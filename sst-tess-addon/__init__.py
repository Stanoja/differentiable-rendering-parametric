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

class SSTTESS_Tessellate(bpy.types.Operator):
	"""My NURBS tessellation addon"""
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
		from .drpg import ParametricPatches, SurfaceTessellator
		from .process_object import add_tessellated_to_blender

		# END CLASSES/utils
		all_start = time()

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

		tess_time = time()
		print(f"Tessellator instantiated in {tess_time - all_start:.3f}s")

		all_patches = []
		for obj in objlist:
			patches = None
			print(f"OBJ {obj.name}")
			try:
				temp_path = Path(tempfile.gettempdir()) / f"{str(obj.name)}.npz"
				patches = ParametricPatches.load(temp_path, device=device)
			except Exception as e:
				print(f"Cannot get patches from obj[{obj.name}] {e}")
				continue
			if patches is not None:
				v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)
				print(f"Number of control points: {len(patches.V)}")
				print(f"Number of patches: {len(patches.F)}")
				print('-TESS Done: {:.3f}s'.format(time() - tess_time))
				add_tessellated_to_blender(v, f)

		print(all_patches)
		for patch in all_patches:
			add_tessellated_to_blender(patch.V, patch.F)
		print('-All Done: {:.3f}s'.format(time() - all_start))
		return {"FINISHED"}

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(self)

class SSTTESS_TessellateBezier(bpy.types.Operator):
	"""My tessellation script"""
	bl_idname = "object.ssttess_tessellate_bezier"
	bl_label = "SST Tessellate object (Bezier)"
	bl_description = "Tessellate selected texture plane objects (Bezier)"
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
		import numpy as np
		import bmesh
		from .drpg import ParametricPatches, SurfaceTessellator

		# TODO SST: Change this
		device = torch.device('cpu')
		from .process_object import add_tessellated_to_blender

		# END CLASSES/utils
		all_start = time()

		objlist = [o for o in context.selected_objects if o.type == 'MESH']
		all_patches = []

		tessellator = SurfaceTessellator(uv_grid_resolution=(self.uv_grid_res_x, self.uv_grid_res_y), type=SurfaceType.Bezier)
		tess_time = time()
		print(f"Tessellator instantiated in {tess_time - all_start:.3f}s")
		import pickle

		print(f"Obj length: {len(objlist)}")
		for obj in objlist:
			patches = None
			print(f"OBJ {obj.name}")
			try:
				temp_path = Path(tempfile.gettempdir()) / f"{str(obj.name)}.npz"
				patches = ParametricPatches.load(temp_path, device=device)
			except Exception as e:
				continue
			if (patches is not None):
				print("BEZIER TESSELATION")
				v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)
				print(f"Number of control points: {len(patches.V)}")
				print(f"Number of patches: {len(patches.F)}")
				print('-BEZ.TESS Done: {:.3f}s'.format(time() - tess_time))
				add_tessellated_to_blender(v, f)

		print('-All Done: {:.3f}s'.format(time() - all_start))
		return {"FINISHED"}

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(self)

class SSTTESS_TessellateCurve(bpy.types.Operator):
	"""My Curve tessellation script"""
	bl_idname = "object.ssttess_tessellate_curve"
	bl_label = "SST Tessellate curve (NURBS)"
	bl_description = "Tessellate selected texture plane objects (NURBS, curves)"
	bl_options = {"REGISTER", "UNDO"}

	curve_resolution: bpy.props.IntProperty(
		name="Curve resolution",
		description="Curve tessellation resolution",
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
		from .drpg import ParametricPatches
		from .drpg import CurveTessellator, circle_profile
		from .process_object import add_tessellated_to_blender

		# END CLASSES/utils
		all_start = time()

		all_object_types = [o.type for o in context.selected_objects]
		objlist = [o for o in context.selected_objects if o.type == 'CURVE']

		import torch
		from pathlib import Path

		profile = circle_profile(0.2)
		tessellator = CurveTessellator((self.curve_resolution))

		all_patches = []
		for obj in objlist:
			curves = None
			print(f"OBJ {obj.name}")
			try:
				temp_path = Path(tempfile.gettempdir()) / f"{str(obj.name)}.npz"
				curves = ParametricPatches.load(temp_path, device=device)
			except Exception as e:
				print(f"Cannot get curves from obj[{obj.name}] {e}")
				continue
			if curves is not None:
				v, f, _, _, _ = tessellator.tessellate(curves=curves, profile=profile)
				print(f"Number of control points: {len(curves.V)}")
				add_tessellated_to_blender(v, f)

		print('-TESS Done: {:.3f}s'.format(time() - all_start))
		print(all_patches)
		for patch in all_patches:
			add_tessellated_to_blender(patch.V, patch.F)
		print('-All Done: {:.3f}s'.format(time() - all_start))
		return {"FINISHED"}

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(self)

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
		default=4,
		min=4,
		max=100
	)

	number_of_control_points: bpy.props.IntProperty(
		name="Number of control points",
		description="Number of control points",
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

		from .drpg import bezier_sphere, ParametricPatches

		device = torch.device("cpu")
		patches = bezier_sphere(num_patches=self.number_of_patches, n=self.number_of_control_points, device=device)

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
		create_blender_mesh_from_patches(patches)
		return {"FINISHED"}

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
		from .process_object import add_tessellated_to_blender

		device = torch.device("cpu")
		n = 5
		patches = bezier_sphere(n=n, device=device, merge_duplicates=False)

		print(f"Number of control points: {len(patches.V)}")
		print(f"Number of patches: {len(patches.F)}")
		tessellator = SurfaceTessellator(uv_grid_resolution=(32, 32),
										 type=SurfaceType.Bezier)  # <- Tessellate as NURBs surface

		v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)
		# reconstructed = ParametricPatches(n=n, V=v, F=f, device=device)

		print(f"{v.shape=}, {f.shape=}")

		add_tessellated_to_blender(v, f)
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
		from .process_object import add_tessellated_to_blender

		device = torch.device("cpu")
		n = 5
		patches = bezier_sphere(n=n, device=device, merge_duplicates=False)

		print(f"Number of control points: {len(patches.V)}")
		print(f"Number of patches: {len(patches.F)}")
		tessellator = SurfaceTessellator(uv_grid_resolution=(32, 32),
										 type=SurfaceType.NURBS)  # <- Tessellate as NURBs surface

		v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)
		# reconstructed = ParametricPatches(n=n, V=v, F=f, device=device)

		print(f"{v.shape=}, {f.shape=}")

		add_tessellated_to_blender(v, f)
		return {"FINISHED"}

class SSTTESS_MultiviewReconstruction(bpy.types.Operator, ImportHelper):
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

class SSTTESS_GenerateBezierHelix(bpy.types.Operator):
	"""My tessellation script - Example Bezier Helix generator"""
	bl_idname = "ssttess.generate_bezier_helix"
	bl_label = "SST Generate Bezier Helix (example)"
	bl_description = "Generates example Bezier Helix"
	bl_options = {"REGISTER", "UNDO"}

	windings: bpy.props.IntProperty(
		name="Number of windings",
		description="Number of windings",
		default=3,
		min=2,
		max=50
	)

	@classmethod
	def poll(cls, context):
		return True

	def execute(self, context):
		import numpy as np
		import torch

		from .drpg import bezier_helix, ParametricPatches, ParametricCurves

		device = torch.device("cpu")
		helix = bezier_helix(windings=self.windings, device=device)
		curves = ParametricCurves(helix.shape[1], 3, device=device)
		curves.add(helix)

		import bpy

		def create_blender_mesh_from_patches(p_curves: ParametricCurves, name="ParamCurveObject"):
			V = p_curves.V
			F = p_curves.F

			# === Create a curve for each entry in F ===
			print(f"V: {V}")
			print(f"F: {F}")
			# for i, f in enumerate(F):
			# 	curve_data = bpy.data.curves.new(name=f"ParamCurve_{i}", type='CURVE')
			# 	curve_data.dimensions = '3D'
			#
			# 	spline = curve_data.splines.new('POLY')  # Use 'BEZIER' or 'NURBS' if needed
			# 	spline.points.add(len(f) - 1)
			#
			# 	for j, idx in enumerate(f):
			# 		x, y, z = V[idx]
			# 		spline.points[j].co = (x, y, z, 1.0)  # 4D: (x, y, z, w)
			#
			# 	# Set display resolution & bevel to make it visible
			# 	curve_data.bevel_depth = 0.01
			# 	curve_data.resolution_u = 12
			#
			# 	# Create object and link to scene
			# 	curve_obj = bpy.data.objects.new(f"ParamCurveObj_{i}", curve_data)
			# 	bpy.context.collection.objects.link(curve_obj)

			curve_data = bpy.data.curves.new(name="ParamCurve", type='CURVE')
			curve_data.dimensions = '3D'
			curve_data.resolution_u = 32
			curve_data.bevel_depth = 0.2  # Make the curves visible

			for i, f in enumerate(F):
				spline = curve_data.splines.new('POLY')
				spline.points.add(len(f) - 1)

				for j, idx in enumerate(f):
					x, y, z = V[idx]
					spline.points[j].co = (x, y, z, 1.0)

			# Create ONE object and link it
			curve_obj = bpy.data.objects.new(name, curve_data)
			bpy.context.collection.objects.link(curve_obj)

			temp_path = Path(tempfile.gettempdir()) / str(name)
			p_curves.save(temp_path)

		print(f"Number of control points: {len(curves.V)}")
		print(f"Number of patches: {len(curves.F)}")

		# Usage:
		# create_bezier_patch_mesh_wireframe(patches)
		create_blender_mesh_from_patches(curves)
		return {"FINISHED"}

	def invoke(self, context, event):
		return context.window_manager.invoke_props_dialog(self)

# REGISTER

def menu_func(self, context):
	self.layout.operator(SSTTESS_Tessellate.bl_idname)
	self.layout.operator(SSTTESS_TessellateCurve.bl_idname)
	self.layout.operator(SSTTESS_GenerateBezierSphere.bl_idname)
	self.layout.operator(SSTTESS_GenerateTessellatedBezierSphere.bl_idname)
	self.layout.operator(SSTTESS_GenerateTessellatedBezierSphereWithNURBS.bl_idname)
	self.layout.operator(SSTTESS_TessellateBezier.bl_idname)
	self.layout.operator(SSTTESS_MultiviewReconstruction.bl_idname)
	self.layout.operator(SSTTESS_GenerateBezierHelix.bl_idname)


classes = (
	SSTTESS_Tessellate,
	SSTTESS_TessellateCurve,
	SSTTESS_InstallModule,
	SSTTESS_AddonPreferences,
	SSTTESS_GenerateBezierSphere,
	SSTTESS_GenerateTessellatedBezierSphere,
	SSTTESS_TessellateBezier,
	SSTTESS_GenerateTessellatedBezierSphereWithNURBS,
	SSTTESS_MultiviewReconstruction,
	SSTTESS_GenerateBezierHelix,
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
