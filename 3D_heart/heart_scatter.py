import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh as stl_mesh

# 加载STL文件
mesh = stl_mesh.Mesh.from_file("heart.stl")

# 提取顶点坐标

vertices_part1 = mesh.vectors[0:2000000:50000].reshape(-1, 3)
vertices_part2 = mesh.vectors[2000000::5000].reshape(-1, 3)

# 绘制顶点
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices_part1[:, 0], vertices_part1[:, 1], vertices_part1[:, 2], c='b', marker='.')
ax.scatter(vertices_part2[:, 0], vertices_part2[:, 1], vertices_part2[:, 2], c='r', marker='.')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
