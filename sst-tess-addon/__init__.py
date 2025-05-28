import tempfile
import json
from array import array
from copyreg import pickle
from pathlib import Path
from enum import Enum, IntEnum
from time import time
from mathutils import Vector, Matrix
import mathutils

# TODO SST: This is a temporary workaround
import sys

sys.path.append("/Users/sstanoje/.local/lib/python3.11/site-packages/NURBSDiff-0.0.0-py3.11-macosx-15.4-arm64.egg")

import bpy
# from bpy.types import Operator
# from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ImportHelper

# PART 1: Setup
bl_info = {
	"name": "SST Tesselation sst-tess-addon",
	"description": "Tessellation sst-tess-addon for diffprog, based on parameterized geometry.",
	"author": "Slobodan Stanojevikj",
	"version": (0, 0, 1),
	"blender": (4, 0, 0),
	"location": "3D view > right toolbar > SST Tessellation sst-tess-addon",
	"category": "Object"
}

# TODO SST: Add the rest of the dependencies
DEPENDENCIES = {
	('torch', 'torch'),
	('typing_extensions', 'typing-extensions'),
	('imageio', 'imageio')
}


def module_can_be_imported(name):
	try:
		__import__(name)
		return True
	except ImportError:
		return False


def get_missing_dependencies():
	return [module for module, _ in DEPENDENCIES if not module_can_be_imported(module)]


# PART 2: DRPG
# TODO SST: Move the classes here

class Side(IntEnum):
	Top = 0
	Left = 1
	Bottom = 2
	Right = 3


class CoordinateMergeMethod(Enum):
	PrimaryPrecedence = 0
	Average = 1


class SurfaceType(Enum):
	Bezier = 0
	NURBS = 1


# PART 4: Blender integration

# TODO SST: Remove if unused
def flatten_face(obj, poly):
	coords = [obj.matrix_world @ obj.data.vertices[i].co for i in poly.vertices]

	if len(coords) < 3:
		raise ValueError("Need at least 3 points to define a plane")

	# Reference point
	v0, v1, v2 = coords[:3]

	# Construct orthonormal basis
	tangent = (v1 - v0).normalized()  # x-axis (local)
	normal = (v1 - v0).cross(v2 - v0).normalized()  # y-axis (local)
	bitangent = normal.cross(tangent).normalized()  # z-axis (local)

	# Project each vertex into local tangent/bitangent (xz-plane)
	flattened = []
	for v in coords:
		rel = v - v0
		x = rel.dot(tangent)
		z = rel.dot(bitangent)  # we call it z now to match the xz-plane
		flattened.append((x, z))

	xs, zs = zip(*flattened)
	# scale = (max(xs) - min(xs), max(zs) - min(zs))  # extent in local xz
	scale = max(max(xs) - min(xs), max(zs) - min(zs)) / 2.0

	# Construct the local-to-world matrix:
	# Grid space: xz-plane, y-up → tangent, bitangent, normal → x, z, y
	local_axes = Matrix((
		tangent,  # x
		normal,  # y
		bitangent  # z
	)).transposed()  # Blender uses column-major matrices

	position = obj.matrix_world @ obj.data.vertices[poly.vertices[0]].co
	return {
		"2d_coords": flattened,
		"normal": normal,
		"tangent": tangent,
		"bitangent": bitangent,
		"scale": scale,
		"position": position,
		"local_to_world": Matrix.Translation(v0) @ local_axes.to_4x4()
	}


class SSTTESS_tessellate(bpy.types.Operator):
	"""My tessellation script"""
	bl_idname = "object.ssttess_tessellate"
	bl_label = "SST Tessellate object (NURBS)"
	bl_description = "Tessellate selected texture plane objects"
	bl_options = {"REGISTER", "UNDO"}

	uv_grid_res_x: bpy.props.IntProperty(
		name="Grid size x (u) ",
		description="Grid size y (u parameter)",
		default=32,
		min=4,
		max=1024
	)
	uv_grid_res_y: bpy.props.IntProperty(
		name="Grid size y (v) ",
		description="Grid size y (v parameter)",
		default=32,
		min=4,
		max=1024
	)

	@classmethod
	def poll(cls, context):
		return context.selected_objects

	def execute(self, context):
		# Classes that depend on torch are imported here, otherwise the plugin fails
		import torch

		# TODO SST: Change this
		device = torch.device('cpu')
		from .process_object import process_object, process_object_2, add_tessellated_to_blender
		from .drpg import ParametricPatches
		from .drpg import bezier_sphere, SurfaceTessellator
		from .process_object import create_tessellated_patches, add_tessellated_to_blender

		# END CLASSES/utils
		all_start = time()

		# ...that have 4 vertices or more
		# objlist = [o for o in bpy.context.selected_objects if o.type == 'MESH' and len(o.data.vertices) >= 4]
		# objlist = [o for o in bpy.context.selected_objects if o.type == 'MESH']
		all_object_types = [o.type for o in context.selected_objects]
		print(f'ALL OBJECT TYPES: {all_object_types}')
		objlist = [o for o in context.selected_objects if o.type == 'MESH']

		uv_grid_res_x = self.uv_grid_res_x
		uv_grid_res_y = self.uv_grid_res_y

		import torch
		import numpy as np
		import bpy
		from pathlib import Path

		tessellator = SurfaceTessellator(uv_grid_resolution=(uv_grid_res_x, uv_grid_res_y),
										 type=SurfaceType.NURBS)  # <- Tessellate as NURBs surface


		def blender_object_to_parametric_patches_dynamic(obj, device: torch.device = None) -> ParametricPatches:
			"""Convert any Blender mesh into ParametricPatches by treating each face as a patch."""
			if obj.type != 'MESH':
				raise ValueError("Object must be a mesh")

			mesh = obj.data
			mesh.calc_loop_triangles()

			patches = ParametricPatches(n=2, m=2, device=device)  # Each face -> 2x2 patch (4 verts)

			for tri in mesh.loop_triangles:
				if len(tri.vertices) != 3:
					continue  # skip non-triangles

				# Get triangle vertex coordinates
				coords = [mesh.vertices[v].co for v in tri.vertices]

				# To form a 2x2 grid (4 points), we can:
				# 1. Place triangle in lower half of a square
				# 2. Add a fake 4th point to complete the square (e.g., duplicate one)

				v0, v1, v2 = coords
				v3 = v2 + (v1 - v0)  # crude extrapolation

				patch_control = torch.tensor(
					[[v0, v1],
					 [v2, v3]],
					dtype=torch.float32,
					device=device
				)

				patches.add(patch_control)

			return patches

		all_patches = []
		for obj in objlist:
			# try:
			# list_all_faces(obj)
			patches = None
			print(f"OBJ {obj.name}")
			try:
				# obj["ssttess"] = pickle.dumps({
				# 	"patches": data,
				# })
				# ssttess = obj["ssttess"]
				# print("FFF")
				# # print(f"SSTESS DATA: {ssttess}")
				# if ssttess is not None:
				# 	print(f"qwe123 {}")
				# 	patches = pickle.loads(ssttess)
				# 	print("qwe234")
				# tmp_patches = ParametricPatches(n=2, m=2, device=device)
				# tmp_patches = ParametricPatches(n=2, m=2, device=device)
				# tmp_patches.load(Path(obj.name), device=device)
				temp_path = Path(tempfile.gettempdir()) / f"{str(obj.name)}.npz"
				# print(f"FFF loading from file: {temp_path}")
				patches = ParametricPatches.load(temp_path, device=device)
			except Exception as e:
				print(f"Cannot get patches from obj[{obj.name}] {e}")
				continue
			if patches is not None:
				v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)
				print(f"Number of control points: {len(patches.V)}")
				print(f"Number of patches: {len(patches.F)}")
				add_tessellated_to_blender(v, f)

			# all_patches.append(blender_object_to_parametric_patches_dynamic(obj, device=device))
			# process_object_2(obj, n = 3, uv_grid_res=(uv_grid_res_x, uv_grid_res_y), device=device)


		print('-All Done: {:.3f}s'.format(time() - all_start))
		print(all_patches)
		for patch in all_patches:
			add_tessellated_to_blender(patch.V, patch.F)
		return {"FINISHED"}

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(self)

class SSTTESS_TessellateBezier(bpy.types.Operator):
	"""My tessellation script"""
	bl_idname = "object.ssttess_tessellate_bezier"
	bl_label = "SST Tessellate object (Bezier)"
	bl_description = "Tessellate selected texture plane objects (Bezier)"
	bl_options = {"REGISTER", "UNDO"}

	@classmethod
	def poll(cls, context):
		return context.selected_objects

	def execute(self, context):
		# Classes that depend on torch are imported here, otherwise the plugin fails
		import torch
		import numpy as np
		import bmesh
		from collections import defaultdict
		from .drpg import ParametricPatches

		# TODO SST: Change this
		device = torch.device('cpu')
		from .process_object import process_object, process_object_2, create_tessellated_patches, mesh_to_parametric_patches, blender_to_parametric_patches

		def create_bezier_patch_mesh(patches, name="BezierPatches"):
			# Convert torch tensors to numpy arrays
			vertices = patches.V.cpu().numpy()
			faces_idx = patches.F.cpu().numpy()

			# Create a new mesh
			mesh = bpy.data.meshes.new(name)

			# For visualization, we'll create:
			# 1. The control points as vertices
			# 2. The control mesh edges
			# 3. The actual patches as faces

			# Create all vertices (control points)
			mesh.vertices.add(len(vertices))
			mesh.vertices.foreach_set("co", vertices.ravel())

			# Create edges for control mesh
			edges = []
			n_patches = faces_idx.shape[0]
			patch_size = faces_idx.shape[1]  # Assuming square patches (3x3)

			for patch_idx in range(n_patches):
				patch_faces = faces_idx[patch_idx]

				# Add horizontal edges
				for row in range(patch_size):
					for col in range(patch_size - 1):
						v1 = patch_faces[row, col]
						v2 = patch_faces[row, col + 1]
						edges.append((v1, v2))

				# Add vertical edges
				for col in range(patch_size):
					for row in range(patch_size - 1):
						v1 = patch_faces[row, col]
						v2 = patch_faces[row + 1, col]
						edges.append((v1, v2))

			# Remove duplicate edges
			edges = list(set(edges))
			mesh.edges.add(len(edges))
			mesh.edges.foreach_set("vertices", np.array(edges).ravel())

			# Update mesh
			mesh.update()

			# Create object and link to scene
			obj = bpy.data.objects.new(name, mesh)
			bpy.context.collection.objects.link(obj)

			return obj

		def mesh_to_patches(obj, n: int, m: int) -> ParametricPatches:
			assert obj.type == 'MESH', "Object must be a mesh"

			mesh = obj.data
			mesh.calc_loop_triangles()

			# Extract all vertices
			vertices = np.array([v.co for v in mesh.vertices], dtype=np.float32)
			V = torch.tensor(vertices)

			num_vertices_per_patch = n * m
			total_vertices = V.shape[0]
			assert total_vertices % num_vertices_per_patch == 0, "Cannot evenly divide vertices into patches"

			# Optional: Store vertex layout (row-major patch grouping)
			num_patches = total_vertices // num_vertices_per_patch

			# Reconstruct dummy face index grid if needed
			# (This assumes you follow the same logic as `generate_grid_faces`)
			# For now we'll skip it

			return ParametricPatches(V, n=n, m=m)

		# END CLASSES/utils
		all_start = time()

		# ...that have 4 vertices or more
		# objlist = [o for o in bpy.context.selected_objects if o.type == 'MESH' and len(o.data.vertices) >= 4]
		# objlist = [o for o in bpy.context.selected_objects if o.type == 'MESH']
		# all_object_types = [o.type for o in context.selected_objects]
		# print(f'ALL OBJECT TYPES: {all_object_types}')
		objlist = [o for o in context.selected_objects if o.type == 'MESH']
		all_patches = []

		n = 3
		uv_grid_res_x = 64
		uv_grid_res_y = 64

		import pickle

		print(f"Obj length: {len(objlist)}")
		for obj in objlist:
			# try:
			# list_all_faces(obj)
			patches = None
			print(f"OBJ {obj.name}")
			try:
				patches = ParametricPatches.load(Path(obj.name), device=device)
			except Exception as e:
				continue
			if (patches is not None):
				print("DO BEZIER TESSELATION")

			# print(f"SST TESSSSS OBJ: {patches}")
			# print(f"\nProcessing object: {obj.name}")
			# print(f"Mesh stats: {len(obj.data.vertices)} vertices, {len(obj.data.polygons)} faces")
			#
			# patches = mesh_to_patches(obj, n=3, m=3)
			# print(f"Patches: {patches}")
			# print(f"Number of control points: {len(patches.V)}")
			# print(f"Number of patches: {len(patches.F)}")
			# create_bezier_patch_mesh(patches, name="BezierPatches_2_0")
			#
			# patches = blender_to_parametric_patches(obj, n=2)
			# print(f"Patches: {patches}")
			# print(f"Number of control points: {len(patches.V)}")
			# print(f"Number of patches: {len(patches.F)}")
			# # create_bezier_patch_mesh(patches, name="BezierPatches_2_1")
			#
			#
			# patches = mesh_to_parametric_patches(obj, n=3, m=3)
			# print(f"Patches: {patches}")
			# print(f"Number of control points: {len(patches.V)}")
			# print(f"Number of patches: {len(patches.F)}")

			# create_bezier_patch_mesh(patches, name="BezierPatches_2_2")

			# except Exception as e:
			# 	print(f"Failed to process {obj.name}: {str(e)}")
			# 	continue

		print('-All Done: {:.3f}s'.format(time() - all_start))
		return {"FINISHED"}

class SSTTESS_InstallModule(bpy.types.Operator):
	bl_idname = "ssttess.install_module"
	bl_label = "Install Module"
	bl_description = "Install the required module"

	package_name: bpy.props.StringProperty(
		name="Package Name",
		description="Name of the package to install"
	)

	def execute(self, context):
		import subprocess
		import sys
		try:
			# Get the user modules path and create it if needed
			user_modules = Path(bpy.utils.user_resource('SCRIPTS', path='modules', create=True))

			# Use subprocess to install the package
			python_exe = Path(sys.executable)
			cmd = [str(python_exe), "-m", "pip", "--no-cache-dir", "install",
				   f"--target={user_modules}", self.package_name, "--no-deps"]

			self.report({'INFO'}, f"Installing {self.package_name}...")

			# Run the installation command
			subprocess.check_call(cmd)

			self.report({'INFO'}, f"Successfully installed {self.package_name}")
			return {'FINISHED'}

		except Exception as e:
			self.report({'ERROR'}, f"Installation failed: {str(e)}")
			return {'CANCELLED'}

class SSTTESS_AddonPreferences(bpy.types.AddonPreferences):
	bl_idname = __package__

	def draw(self, context):
		layout = self.layout

		layout.label(text="Required Dependencies:")

		box = layout.box()
		col = box.column()

		row = col.row()
		has_torch = module_can_be_imported('torch')
		if has_torch:
			row.label(text="Torch: Installed", icon="CHECKMARK")
		else:
			row.label(text="Torch: Not Installed", icon="ERROR")
			row.operator("ssttess.install_module", text="Install torch").package_name = "torch"

		has_typing = module_can_be_imported('typing_extensions')
		row = col.row()
		if has_typing:
			row.label(text="Typing Extensions: Installed", icon="CHECKMARK")
		else:
			row.label(text="Typing Extensions: Not Installed", icon="ERROR")
			row.operator("ssttess.install_module", text="Install typing extensions").package_name = "typing-extensions"

		has_imageio = module_can_be_imported('imageio')
		row = col.row()
		if has_imageio:
			row.label(text="Imageio: Installed", icon="CHECKMARK")
		else:
			row.label(text="Imageio: Not Installed", icon="ERROR")
			row.operator("ssttess.install_module", text="Install Imageio").package_name = "imageio"

		# Help text
		layout.separator()
		col = layout.column()
		col.label(text="If installation fails, try running Blender as administrator")
		col.label(text="or install packages manually into your Blender modules folder:")
		col.label(text=str(Path(bpy.utils.user_resource('SCRIPTS', path='modules'))))

class SSTTESS_GenerateBezierSphere(bpy.types.Operator):
	"""My tessellation script - Example Bezier sphere generator"""
	bl_idname = "ssttess.generate_bezier_sphere"
	bl_label = "SST Generate Bezier Sphere (example)"
	bl_description = "Generates example Bezier Sphere"
	bl_options = {"REGISTER", "UNDO"}

	number_of_patches: bpy.props.IntProperty(
		name="Number of patches",
		description="Number of patches",
		default=5,
		min=4,
		max=100
	)

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		import numpy as np
		import torch

		from .drpg import bezier_sphere, generate_grid_faces, ParametricPatches

		device = torch.device("cpu")
		patches = bezier_sphere(n=self.number_of_patches, device=device)

		def create_bezier_patch_mesh_wireframe(patches, name="BezierPatches"):
			# Convert torch tensors to numpy arrays
			vertices = patches.V.cpu().numpy()
			faces_idx = patches.F.cpu().numpy()

			# Create a new mesh
			mesh = bpy.data.meshes.new(name)

			# For visualization, we'll create:
			# 1. The control points as vertices
			# 2. The control mesh edges
			# 3. The actual patches as faces

			# Create all vertices (control points)
			mesh.vertices.add(len(vertices))
			mesh.vertices.foreach_set("co", vertices.ravel())

			# Create edges for control mesh
			edges = []
			n_patches = faces_idx.shape[0]
			patch_size = faces_idx.shape[1]  # Assuming square patches (3x3)

			for patch_idx in range(n_patches):
				patch_faces = faces_idx[patch_idx]

				# Add horizontal edges
				for row in range(patch_size):
					for col in range(patch_size - 1):
						v1 = patch_faces[row, col]
						v2 = patch_faces[row, col + 1]
						edges.append((v1, v2))

				# Add vertical edges
				for col in range(patch_size):
					for row in range(patch_size - 1):
						v1 = patch_faces[row, col]
						v2 = patch_faces[row + 1, col]
						edges.append((v1, v2))

			# Remove duplicate edges
			edges = list(set(edges))
			mesh.edges.add(len(edges))
			mesh.edges.foreach_set("vertices", np.array(edges).ravel())

			# Update mesh
			mesh.update()

			# Create object and link to scene
			obj = bpy.data.objects.new(name, mesh)

			# TODO SST: Attach the patches on the object
			import pickle

			data = {
				'n': patches.n,
				'm': patches.m,
				'V': patches.V.cpu().detach().numpy(),
				'F': patches.F.cpu().detach().numpy(),
			}
			obj["ssttess"] = pickle.dumps({
				"patches": data,
			})
			print(f"FFF: {obj['ssttess']}")
			bpy.context.collection.objects.link(obj)

			return obj

		import bpy
		import bmesh

		def create_blender_mesh_from_patches(patches: ParametricPatches, name="ParametricPatchObject"):
			# Convert to CPU numpy arrays
			V = patches.V.cpu().detach().numpy()
			F = patches.F.cpu().detach().numpy()

			# Flatten all patches into quads or triangles
			verts = V.tolist()
			faces = []

			for patch in F:
				for i in range(patch.shape[0] - 1):
					for j in range(patch.shape[1] - 1):
						v0 = int(patch[i][j])
						v1 = int(patch[i][j + 1])
						v2 = int(patch[i + 1][j + 1])
						v3 = int(patch[i + 1][j])
						faces.append([v0, v1, v2, v3])  # quad face

			# Create a new mesh and object
			mesh = bpy.data.meshes.new(name + "_Mesh")
			mesh.from_pydata(verts, [], faces)
			mesh.update()

			obj = bpy.data.objects.new(name, mesh)
			import pickle

			data = {
				'n': patches.n,
				'm': patches.m,
				'V': patches.V.cpu().detach().numpy(),
				'F': patches.F.cpu().detach().numpy(),
			}
			obj["ssttess"] = pickle.dumps({
				"patches": data,
			})
			temp_path = Path(tempfile.gettempdir()) / str(obj.name)
			patches.save(temp_path)
			# print(f"FFF saved to file: {temp_path}")
			bpy.context.collection.objects.link(obj)

			return obj


		print(f"Number of control points: {len(patches.V)}")
		print(f"Number of patches: {len(patches.F)}")
		print(f"Number of n: {patches.n} m: {patches.m}")

		# Usage:
		# create_bezier_patch_mesh_wireframe(patches)
		create_blender_mesh_from_patches(patches)
		return {"FINISHED"}
		# np.random.seed(913)
		# color = (np.random.rand(patches.F.shape[0], 3) * 255).astype(np.uint8)

		# C = patches.V[patches.F]
		# for i in range(patches.F.shape[0]):
		# 	c = C[i]
		#
		# 	# Create geometry for the control mesh
		# 	f = generate_grid_faces(patches.n, patches.m)

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(self)

class SSTTESS_GenerateTessellatedBezierSphere(bpy.types.Operator):
	"""My tessellation script - Example Tessellated Bezier sphere generator"""
	bl_idname = "ssttess.generate_tessellated_bezier_sphere"
	bl_label = "SST Generate Tessellated Bezier Sphere (example)"
	bl_description = "Generates example tessellated Bezier Sphere"
	bl_options = {"REGISTER", "UNDO"}

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		import torch

		from .drpg import bezier_sphere, SurfaceTessellator
		from .process_object import create_tessellated_patches, add_tessellated_to_blender

		device = torch.device("cpu")
		n = 5
		patches = bezier_sphere(n=n, device=device, merge_duplicates=False)

		# def evaluate_bezier_patch(control_points, resolution=10):
		# 	"""Evaluate a Bézier patch at given resolution"""
		# 	# control_points should be a 3x3x3 array
		# 	u = np.linspace(0, 1, resolution)
		# 	v = np.linspace(0, 1, resolution)
		#
		# 	# Bernstein basis functions
		# 	B = lambda i, n, t: comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
		#
		# 	points = []
		# 	for ui in u:
		# 		for vi in v:
		# 			p = np.zeros(3)
		# 			for i in range(3):
		# 				for j in range(3):
		# 					p += B(i, 2, ui) * B(j, 2, vi) * control_points[i, j]
		# 			points.append(p)
		#
		# 	# Create faces
		# 	faces = []
		# 	for i in range(resolution - 1):
		# 		for j in range(resolution - 1):
		# 			v1 = i * resolution + j
		# 			v2 = v1 + 1
		# 			v3 = (i + 1) * resolution + j + 1
		# 			v4 = (i + 1) * resolution + j
		# 			faces.append((v1, v2, v3, v4))
		#
		# 	return np.array(points), np.array(faces)

		print(f"Number of control points: {len(patches.V)}")
		print(f"Number of patches: {len(patches.F)}")
		tessellator = SurfaceTessellator(uv_grid_resolution=(32, 32),
										 type=SurfaceType.Bezier)  # <- Tessellate as NURBs surface

		v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)
		# reconstructed = ParametricPatches(n=n, V=v, F=f, device=device)

		print(f"{v.shape=}, {f.shape=}")

		name = "TessObj"

		add_tessellated_to_blender(v, f)
		# create_tessellated_patches(patches, resolution=10)
		# create_tessellated_patches(reconstructed, resolution=10)
		return {"FINISHED"}

class SSTTESS_GenerateTessellatedBezierSphereWithNURBS(bpy.types.Operator):
	"""My tessellation script - Example Tessellated Bezier sphere generator, tessellated with NURBS"""
	bl_idname = "ssttess.generate_tessellated_bezier_sphere_nurbs_example"
	bl_label = "SST Generate Tessellated Bezier Sphere (NURBS example)"
	bl_description = "Generates example tessellated Bezier Sphere with NURBS"
	bl_options = {"REGISTER", "UNDO"}

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		import torch

		from .drpg import bezier_sphere, SurfaceTessellator
		from .process_object import create_tessellated_patches, add_tessellated_to_blender

		device = torch.device("cpu")
		n = 5
		patches = bezier_sphere(n=n, device=device, merge_duplicates=False)

		# def evaluate_bezier_patch(control_points, resolution=10):
		# 	"""Evaluate a Bézier patch at given resolution"""
		# 	# control_points should be a 3x3x3 array
		# 	u = np.linspace(0, 1, resolution)
		# 	v = np.linspace(0, 1, resolution)
		#
		# 	# Bernstein basis functions
		# 	B = lambda i, n, t: comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
		#
		# 	points = []
		# 	for ui in u:
		# 		for vi in v:
		# 			p = np.zeros(3)
		# 			for i in range(3):
		# 				for j in range(3):
		# 					p += B(i, 2, ui) * B(j, 2, vi) * control_points[i, j]
		# 			points.append(p)
		#
		# 	# Create faces
		# 	faces = []
		# 	for i in range(resolution - 1):
		# 		for j in range(resolution - 1):
		# 			v1 = i * resolution + j
		# 			v2 = v1 + 1
		# 			v3 = (i + 1) * resolution + j + 1
		# 			v4 = (i + 1) * resolution + j
		# 			faces.append((v1, v2, v3, v4))
		#
		# 	return np.array(points), np.array(faces)

		print(f"Number of control points: {len(patches.V)}")
		print(f"Number of patches: {len(patches.F)}")
		tessellator = SurfaceTessellator(uv_grid_resolution=(32, 32),
										 type=SurfaceType.NURBS)  # <- Tessellate as NURBs surface

		v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)
		# reconstructed = ParametricPatches(n=n, V=v, F=f, device=device)

		print(f"{v.shape=}, {f.shape=}")

		name = "TessObj"

		add_tessellated_to_blender(v, f)
		# create_tessellated_patches(patches, resolution=10)
		# create_tessellated_patches(reconstructed, resolution=10)
		return {"FINISHED"}

class SSTTESS_ImageReconstruction(bpy.types.Operator, ImportHelper):
	"""Prompt the user to select images for image reconstruction."""
	bl_idname = "ssttess.multiview_reconstruction"
	bl_label = "SST Multiview reconstruction"

	filter_glob: bpy.props.StringProperty(
		default="*.png",
		options={'HIDDEN'},
		maxlen=255,
	)
	files: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)

	def execute(self, context):
		directory = self.filepath.rsplit("/", 1)[0]
		selected_files = [directory + "/" + f.name for f in self.files]
		print("Selected files:", selected_files)
		# call the Multiview Reconstruction properly
		return {'FINISHED'}


# REGISTER

def menu_func(self, context):
	self.layout.operator(SSTTESS_tessellate.bl_idname)
	self.layout.operator(SSTTESS_GenerateBezierSphere.bl_idname)
	self.layout.operator(SSTTESS_GenerateTessellatedBezierSphere.bl_idname)
	self.layout.operator(SSTTESS_GenerateTessellatedBezierSphereWithNURBS.bl_idname)
	self.layout.operator(SSTTESS_TessellateBezier.bl_idname)
	self.layout.operator(SSTTESS_ImageReconstruction.bl_idname)


classes = (
	SSTTESS_tessellate,
	SSTTESS_InstallModule,
	SSTTESS_AddonPreferences,
	SSTTESS_GenerateBezierSphere,
	SSTTESS_GenerateTessellatedBezierSphere,
	SSTTESS_TessellateBezier,
	SSTTESS_GenerateTessellatedBezierSphereWithNURBS,
	SSTTESS_ImageReconstruction,
)


def register():
	for cls in classes:
		bpy.utils.register_class(cls)
	bpy.types.VIEW3D_MT_object.append(menu_func)
	print("SST Tessellation sst-tess-addon registered")


def unregister():
	for cls in reversed(classes):
		bpy.utils.unregister_class(cls)
	print("SST Tessellation sst-tess-addon unregistered")


if __name__ == "__main__":
	register()
