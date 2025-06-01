import bpy
from .drpg.tessellation import SurfaceTessellator, SurfaceType
from .drpg.bezier_surface import get_normals_from_tangents
from .drpg.spline_surface import ParametricPatches
from .drpg.geometry import create_planar_grid
import torch
import numpy as np
import mathutils
import bmesh
from math import comb  # Built-in in Python 3.10+
from collections import defaultdict

# TODO SST: Move the rest separately to adapter package
def generate_faces(n_rows, n_cols):
	faces = []
	for i in range(n_rows - 1):
		for j in range(n_cols - 1):
			a = i * n_cols + j
			b = a + 1
			c = a + n_cols
			d = c + 1
			# Two triangles per quad
			faces.append([a, b, c])
			faces.append([b, d, c])
	return torch.tensor(faces, dtype=torch.int64)


def detach_matrix(mm):
	if hasattr(mm, 'detach'):
		return mm.detach().cpu().numpy().reshape(-1, 3)
	return mm.reshape(-1, 3)


def create_patch_mesh(name, verts, faces, collection=None):
	# Convert PyTorch tensors to NumPy
	verts_np = detach_matrix(verts)
	faces_np = detach_matrix(faces)

	# Safety checks
	if not np.isfinite(verts_np).all():
		raise ValueError(f"{name}: Vertices contain NaNs or Infs")

	if (faces_np < 0).any() or (faces_np >= verts_np.shape[0]).any():
		raise ValueError(f"{name}: Face indices out of bounds")

	# Create Blender mesh
	mesh = bpy.data.meshes.new(name)
	obj = bpy.data.objects.new(name, mesh)
	(collection or bpy.context.collection).objects.link(obj)

	# Build the mesh
	verts_list = verts_np.tolist()
	faces_list = faces_np.astype(np.int32).tolist()

	mesh.from_pydata(verts_list, [], faces_list)
	mesh.validate()
	mesh.update()
	return obj


def create_curved_grid(n, device=None):
	u = torch.linspace(-1, 1, n, device=device)
	v = torch.linspace(-1, 1, n, device=device)
	uu, vv = torch.meshgrid(u, v, indexing='ij')
	# Example: Paraboloid surface
	zz = uu ** 2 + vv ** 2
	grid = torch.stack([uu, vv, zz], dim=-1)
	return grid


def compute_face_transform(obj, poly):
	# Get the world matrix of the object
	world_matrix = obj.matrix_world

	# Compute the face center in world coordinates
	face_center = world_matrix @ poly.center

	# Compute the face normal in world coordinates
	face_normal = world_matrix.to_3x3() @ poly.normal

	# face_normal = -face_normal

	# Choose an arbitrary vector not parallel to the normal
	arbitrary = mathutils.Vector((0, 0, 1))
	if abs(face_normal.dot(arbitrary)) > 0.999:
		arbitrary = mathutils.Vector((1, 0, 0))

	# Compute the tangent and bitangent vectors
	tangent = face_normal.cross(arbitrary).normalized()
	bitangent = face_normal.cross(tangent).normalized()

	# Create a 3x3 rotation matrix
	rotation = mathutils.Matrix((tangent, bitangent, face_normal)).transposed()

	# Combine rotation and translation into a 4x4 matrix
	transform = rotation.to_4x4()
	transform.translation = face_center

	return transform


def compute_face_transform_2(obj, bm_face):
	"""Compute transformation matrix for a BMesh face"""
	from mathutils import Vector, Matrix

	world_matrix = obj.matrix_world
	inv_world_matrix = world_matrix.inverted()

	# Calculate face center (median of vertices)
	face_center = Vector()
	for vert in bm_face.verts:
		face_center += vert.co
	face_center /= len(bm_face.verts)

	# Transform to world space
	face_center = world_matrix @ face_center
	face_normal = (world_matrix.to_3x3() @ bm_face.normal).normalized()

	# Calculate tangent using first edge
	if len(bm_face.verts) >= 2:
		edge_vec = (bm_face.verts[1].co - bm_face.verts[0].co).normalized()
	else:
		edge_vec = Vector((1, 0, 0))

	# Create orthogonal coordinate system
	tangent = edge_vec
	bitangent = face_normal.cross(tangent).normalized()

	# Handle degenerate cases
	if bitangent.length < 0.001:
		tangent = Vector((0, 1, 0)) if abs(face_normal.z) > 0.9 else Vector((0, 0, 1))
		bitangent = face_normal.cross(tangent).normalized()

	# Build transformation matrix
	rot_matrix = Matrix()
	rot_matrix.col[0] = tangent
	rot_matrix.col[1] = bitangent
	rot_matrix.col[2] = face_normal

	transform = rot_matrix.to_4x4()
	transform.translation = face_center

	return transform

def apply_transform_to_patch(verts, transform, device=None):
	"""More efficient transformation using PyTorch"""
	import torch

	# Convert transform to PyTorch tensor
	transform_t = torch.tensor(np.array(transform), dtype=torch.float32, device=device)

	# Add homogeneous coordinate
	ones = torch.ones((*verts.shape[:-1], 1), device=device)
	verts_homo = torch.cat([verts, ones], dim=-1)

	# Apply transform (batch matrix multiplication)
	transformed = torch.einsum('...ij,...j->...i', transform_t, verts_homo)
	return transformed[..., :3]


def evaluate_bezier_patch(control_points, resolution=10):
	"""Evaluate a 5×5 Bézier patch"""
	u = np.linspace(0, 1, resolution)
	v = np.linspace(0, 1, resolution)

	# Quintic Bernstein basis
	B = lambda i, n, t: comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

	points = np.zeros((resolution, resolution, 3))
	for i, ui in enumerate(u):
		for j, vi in enumerate(v):
			p = np.zeros(3)
			for m in range(5):  # 5 control points in u direction
				for n in range(5):  # 5 control points in v direction
					p += B(m, 4, ui) * B(n, 4, vi) * control_points[m, n]
			points[i, j] = p

	# Create faces
	faces = []
	for i in range(resolution - 1):
		for j in range(resolution - 1):
			v1 = i * resolution + j
			v2 = v1 + 1
			v3 = (i + 1) * resolution + j + 1
			v4 = (i + 1) * resolution + j
			faces.append([v1, v2, v3, v4])

	return points.reshape(-1, 3), np.array(faces)


def create_tessellated_patches(patches, resolution=10, name="TessellatedPatches"):
	all_verts = []
	all_faces = []
	vert_offset = 0

	# Convert to numpy and reshape to (n_patches, 5, 5, 3)
	C = patches.V[patches.F].cpu().numpy()

	for patch_idx in range(C.shape[0]):
		# Get 5×5 control points for this patch
		control_points = C[patch_idx].reshape(5, 5, 3)
		verts, faces = evaluate_bezier_patch(control_points, resolution)

		all_verts.append(verts)
		all_faces.append(faces + vert_offset)
		vert_offset += len(verts)

	# Combine and triangulate
	verts_array = np.concatenate(all_verts)
	tri_faces = []
	for quad in np.concatenate(all_faces):
		tri_faces.append([quad[0], quad[1], quad[2]])
		tri_faces.append([quad[0], quad[2], quad[3]])

	# Create mesh
	mesh = bpy.data.meshes.new(name)
	mesh.from_pydata(verts_array, [], tri_faces)
	mesh.update()

	obj = bpy.data.objects.new(name, mesh)
	bpy.context.collection.objects.link(obj)
	return obj


def mesh_to_parametric_patches_old(obj, n=3, m=None, device=None):
	"""
	Convert a Blender mesh object to ParametricPatches while preserving original patch structure.

	Args:
		obj: Blender mesh object with pre-generated patches
		n: Number of control points in vertical direction (default=3)
		m: Number of control points in horizontal direction (default=None, same as n)
		device: Torch device for the output tensors

	Returns:
		ParametricPatches object containing the original patch structure
	"""
	if m is None:
		m = n

	# Initialize patches object
	patches = ParametricPatches(n, m, device=device)

	# Get mesh data
	mesh = obj.data
	bm = bmesh.new()
	bm.from_mesh(mesh)
	bm.verts.ensure_lookup_table()

	# Extract all vertex positions
	V = torch.tensor([v.co for v in bm.verts], dtype=torch.float32, device=device)
	patches.V = V  # Assign all vertices at once

	# Check for custom properties storing patch information
	if "num_patches" in mesh:
		num_patches = mesh["num_patches"]
		print(f"Found {num_patches} patches in custom properties")
	else:
		# Estimate number of patches based on vertex count
		num_patches = len(bm.verts) // (n * m)
		print(f"Estimated {num_patches} patches from vertex count")

	# Reconstruct patch indices
	if num_patches > 0:
		# Create patch indices assuming vertices are ordered by patch
		total_control_points = num_patches * n * m
		if len(bm.verts) < total_control_points:
			raise ValueError(f"Not enough vertices ({len(bm.verts)}) for {num_patches} patches of size {n}x{m}")

		# Create index array for all patches
		F = torch.arange(total_control_points, device=device).reshape(num_patches, n, m)
		patches.F = F

		# Verify we have the expected structure
		print(f"Created {len(patches.F)} patches with {len(patches.V)} control points")
		print(f"First patch indices:\n{patches.F[0]}")
	else:
		raise ValueError("No patches found in mesh")

	bm.free()
	return patches

def mesh_to_parametric_patches(obj, n=3, m=None, device=None):
	"""
	Convert all mesh vertices into a single parametric patch containing all vertices.

	Args:
		obj: Blender mesh object
		n: Control points in vertical direction
		m: Control points in horizontal direction
		device: Torch device for output

	Returns:
		ParametricPatches with all vertices in one patch
	"""
	if m is None:
		m = n

	# Initialize patches object
	patches = ParametricPatches(n, m, device=device)

	# Get all vertices
	bm = bmesh.new()
	bm.from_mesh(obj.data)
	bm.verts.ensure_lookup_table()

	# Store vertex positions
	patches.V = torch.tensor([v.co for v in bm.verts], dtype=torch.float32, device=device)

	# Create a single patch containing all vertices
	num_vertices = len(bm.verts)

	# Calculate how many vertices we can fit in n x m grid
	max_vertices = n * m
	if num_vertices > max_vertices:
		print(f"Warning: Using first {max_vertices} vertices of {num_vertices}")
		vertex_indices = torch.arange(max_vertices, device=device)
	else:
		# Pad with last vertex if needed
		vertex_indices = torch.arange(num_vertices, device=device)
		if num_vertices < max_vertices:
			padding = torch.full((max_vertices - num_vertices,),
								 vertex_indices[-1],
								 device=device)
			vertex_indices = torch.cat([vertex_indices, padding])

	# Create single patch containing (up to) n x m vertices
	patches.F = vertex_indices.reshape(1, n, m)

	bm.free()

	print(f"Created 1 patch containing {len(patches.V)} vertices")
	print("Patch vertex indices:")
	print(patches.F[0].cpu().numpy())

	return patches


def blender_to_parametric_patches(obj, n: int, m: int = None, device='cpu') -> ParametricPatches:
	""" Convert a Blender mesh object (with or without faces) to ParametricPatches """

	m = m if m is not None else n
	mesh = obj.to_mesh(preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get())
	mesh.calc_loop_triangles()

	patches = ParametricPatches(n=n, m=m, device=torch.device(device))

	vertices = [v.co.copy() for v in mesh.vertices]
	verts_np = np.array([v[:] for v in vertices], dtype=np.float32)

	if len(mesh.polygons) > 0:
		# Mesh has faces → create patches based on quads
		for poly in mesh.polygons:
			if len(poly.vertices) == n * m:
				patch_verts = [verts_np[idx] for idx in poly.vertices]
				patch_grid = np.array(patch_verts, dtype=np.float32).reshape((n, m, 3))
				patches.add(torch.tensor(patch_grid, dtype=torch.float32))
			elif len(poly.vertices) == 4 and n == 2 and m == 2:
				# Automatically treat quads as 2x2 patches
				patch_verts = [verts_np[idx] for idx in poly.vertices]
				patch_grid = np.array(patch_verts, dtype=np.float32).reshape((2, 2, 3))
				patches.add(torch.tensor(patch_grid, dtype=torch.float32))
			else:
				print(f"Skipped polygon with {len(poly.vertices)} vertices.")
	else:
		# No faces → wireframe. Try to extract grid-style patches from edge layout
		edge_map = {}
		for e in mesh.edges:
			v1, v2 = e.vertices
			edge_map.setdefault(v1, []).append(v2)
			edge_map.setdefault(v2, []).append(v1)

		visited = set()
		for i, vi in enumerate(mesh.vertices):
			if i in visited:
				continue
			# Attempt to construct a (n x m) grid patch from neighbors
			# Very basic heuristic: look for a row-major grid using BFS
			row = [i]
			curr = i
			for _ in range(m - 1):
				nexts = [v for v in edge_map.get(curr, []) if v not in row]
				if not nexts: break
				curr = nexts[0]
				row.append(curr)
			if len(row) < m:
				continue

			grid = [row]
			for _ in range(n - 1):
				next_row = []
				for v in grid[-1]:
					nbrs = [vn for vn in edge_map.get(v, []) if vn not in visited and vn not in sum(grid, [])]
					if not nbrs: break
					next_row.append(nbrs[0])
				if len(next_row) < m:
					break
				grid.append(next_row)

			if len(grid) == n:
				coords = np.array([[verts_np[v] for v in row] for row in grid], dtype=np.float32)
				patches.add(torch.tensor(coords, dtype=torch.float32))
				visited.update(sum(grid, []))

	return patches


def add_tessellated_to_blender(v: torch.Tensor, f: torch.Tensor, name="TessellatedObject"):
	"""
	v: torch.Tensor of shape (N, H, W, 3)
	f: torch.Tensor of shape (N, F, 3)
	"""

	all_vertices = []
	all_faces = []
	vertex_offset = 0

	N = v.shape[0]  # number of patches

	for i in range(v.shape[0]):
		verts = v[i].reshape(-1, 3).cpu().numpy().tolist()
		faces = f[i].reshape(-1, 3).cpu().numpy().tolist()

		all_vertices.extend(verts)

		# IMPORTANT: Do NOT add offset here if faces are global indices
		all_faces.extend(faces)

		vertex_offset += len(verts)  # can still increment vertex_offset to check counts

	# Sanity check
	max_face_index = max(max(face) for face in all_faces)
	if max_face_index >= len(all_vertices):
		raise ValueError(f"Corrupted mesh: max face index {max_face_index} >= {len(all_vertices)} vertices")

	# Create Blender mesh
	mesh = bpy.data.meshes.new(name + "_mesh")
	mesh.from_pydata(all_vertices, [], all_faces)
	mesh.validate()
	mesh.update()

	obj = bpy.data.objects.new(name, mesh)
	bpy.context.collection.objects.link(obj)
	return obj


def create_bezier_patch_mesh_wireframe(patches, name="BezierPatches"):
	vertices = patches.V.cpu().numpy()
	faces_idx = patches.F.cpu().numpy()

	mesh = bpy.data.meshes.new(name)
	# For visualization, we'll create:
	# 1. The control points as vertices
	# 2. The control mesh edges
	# 3. The actual patches as faces


	# Control points
	mesh.vertices.add(len(vertices))
	mesh.vertices.foreach_set("co", vertices.ravel())

	# Edges for control mesh
	edges = []
	n_patches = faces_idx.shape[0]
	patch_size = faces_idx.shape[1]  # Assuming square patches (3x3)

	for patch_idx in range(n_patches):
		patch_faces = faces_idx[patch_idx]

		# Horizontal edges
		for row in range(patch_size):
			for col in range(patch_size - 1):
				v1 = patch_faces[row, col]
				v2 = patch_faces[row, col + 1]
				edges.append((v1, v2))

		# Vertical edges
		for col in range(patch_size):
			for row in range(patch_size - 1):
				v1 = patch_faces[row, col]
				v2 = patch_faces[row + 1, col]
				edges.append((v1, v2))

	# Remove duplicate edges, same as in their library (still under testing whether it is fully needed)
	edges = list(set(edges))
	mesh.edges.add(len(edges))
	mesh.edges.foreach_set("vertices", np.array(edges).ravel())

	# Update mesh
	mesh.update()

	# Create object and link to scene
	obj = bpy.data.objects.new(name, mesh)

	# NOTE SST: Attach the patches on the object
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


def process_object(obj, n=3, uv_grid_res=(64, 64), device=None):
	if obj.type != 'MESH':
		return None

	# Create a BMesh for reliable face access
	bm = bmesh.new()
	bm.from_mesh(obj.data)
	bm.faces.ensure_lookup_table()

	# Initialize patches and collect transformations
	patches = ParametricPatches(n, device)
	transforms = []

	for poly in obj.data.polygons:
		grid = create_curved_grid(n, device)
		patches.add(grid)
		transforms.append(compute_face_transform(obj, poly))

	# Tessellate all patches at once
	tessellator = SurfaceTessellator(
		uv_grid_resolution=uv_grid_res,
		type=SurfaceType.NURBS
	)
	v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)

	# Combine all patches into a single mesh
	all_verts = []
	all_faces = []
	vertex_offset = 0

	for i, (verts, faces) in enumerate(zip(v, f)):
		transformed_verts = apply_transform_to_patch(verts, transforms[i])
		all_verts.append(transformed_verts.reshape(-1, 3))

		# Adjust face indices with current offset
		adjusted_faces = faces - faces.min() + vertex_offset
		all_faces.append(adjusted_faces)

		vertex_offset += len(transformed_verts)

	# Create single combined mesh
	if all_verts:
		combined_verts = np.concatenate(all_verts, axis=0)
		combined_faces = np.concatenate(all_faces, axis=0)
		return create_patch_mesh(f"{obj.name}_Tessellated", combined_verts, combined_faces)
	return None


def process_object_2(obj, n=3, uv_grid_res=(64, 64), device=None):
	if obj.type == 'MESH':
		patches = ParametricPatches(n, device=None)
		for poly in obj.data.polygons:
			# Generate the patch grid (ensure it's curved if needed)
			grid = create_curved_grid(n, device=None)
			patches.add(grid)

		tessellator = SurfaceTessellator(
			uv_grid_resolution=uv_grid_res, type=SurfaceType.NURBS
		)
		v, f, tu, tv = tessellator.tessellate(patches, degree=2, return_per_patch=True)

		for i, poly in enumerate(obj.data.polygons):
			transform = compute_face_transform(obj, poly)
			transformed_verts = apply_transform_to_patch(v[i], transform)
			faces_i = f[i] - f[i].min()
			create_patch_mesh(f"Patch_{i}", transformed_verts, faces_i)

