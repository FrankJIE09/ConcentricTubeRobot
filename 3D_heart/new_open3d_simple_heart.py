import open3d as o3d

# 加载STL文件
mesh_in = o3d.io.read_triangle_mesh("heart.stl")
mesh_in.compute_vertex_normals()
print(
    f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
)
# o3d.visualization.draw_geometries([mesh_in])

voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 320
print(f'voxel_size = {voxel_size:e}')
mesh_smp = mesh_in.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Average)
print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)
# o3d.visualization.draw_geometries([mesh_smp])
mesh_smp.compute_vertex_normals()
o3d.io.write_triangle_mesh("simplified_mesh.stl", mesh_smp)