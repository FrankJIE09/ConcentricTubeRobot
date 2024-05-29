import numpy as np
from stl import mesh
begin_node_number = 2850000
end_node_number = -1
gap = 10

part_mod = False
if part_mod is True:
    gap_mod = False
else:
    gap_mod = True
# Load the STL file
your_mesh = mesh.Mesh.from_file('../simplified_mesh.stl')

# Extract vertices and faces
vertices = your_mesh.vectors

# Simplify the mesh (reduce number of vertices)
if part_mod is True:
    flat_reduced_vertices = vertices[begin_node_number:end_node_number]
    new_mesh = mesh.Mesh(np.zeros(len(flat_reduced_vertices), dtype=mesh.Mesh.dtype))
    for i, vertex in enumerate(flat_reduced_vertices):
        new_mesh.vectors[i] = vertex
    new_mesh.save('small_heart_' + str(begin_node_number) + '-' + str(end_node_number) + '.stl')

if gap_mod is True:
    flat_reduced_vertices = np.vstack((vertices[0:200000:gap*50],vertices[200000:-10000:gap]))

    new_mesh = mesh.Mesh(np.zeros(len(flat_reduced_vertices), dtype=mesh.Mesh.dtype))
    for i, vertex in enumerate(flat_reduced_vertices):
        new_mesh.vectors[i] = vertex
    new_mesh.save('small_heart_gap' + str(gap) + '.stl')
