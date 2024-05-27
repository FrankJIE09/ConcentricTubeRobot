import numpy as np

# 定义初始状态的切线向量和导数
def initial_tangent_vector(s):
    # 这里只是一个示例，可以根据具体情况修改
    return np.array([1, 0])

def derivative_initial_tangent_vector(s):
    # 这里只是一个示例，可以根据具体情况修改
    return np.array([0, 1])

# 定义变形状态的切线向量和导数
def deformed_tangent_vector(s):
    # 这里只是一个示例，可以根据具体情况修改
    return np.array([1, np.sin(s)])

def derivative_deformed_tangent_vector(s):
    # 这里只是一个示例，可以根据具体情况修改
    return np.array([0, np.cos(s)])

# 计算曲率向量
def curvature_vector(s):
    tangent = deformed_tangent_vector(s)
    derivative_tangent = derivative_deformed_tangent_vector(s)
    curvature = np.cross(tangent, derivative_tangent) / np.linalg.norm(tangent)**3
    return curvature

# 计算扭转向量
def torsion_vector(s):
    tangent = deformed_tangent_vector(s)
    derivative_tangent = derivative_deformed_tangent_vector(s)
    torsion = np.dot(tangent, derivative_tangent) / np.linalg.norm(tangent)**2
    return torsion

# 在特定点处计算曲率和扭转向量
s = np.pi / 2  # 假设曲线参数为 pi/2
curvature = curvature_vector(s)
torsion = torsion_vector(s)
print("曲率向量：", curvature)
print("扭转向量：", torsion)
