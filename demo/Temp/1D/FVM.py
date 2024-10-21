import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 0.5  # 杆的长度 (m)
k = 1000  # 导热系数 (W/m·K)
A = 10e-3  # 截面积 (m^2)

# 边界条件
T_A = 100  # 左边界温度 (°C)
T_B = 500  # 右边界温度 (°C)

# 离散化参数
N = 5  # 控制体的数量
dx = L / (N + 1)  # 每个控制体的长度

# 初始化系数矩阵和右端项
A_matrix = np.zeros((N, N))
b = np.zeros(N)

# 填充系数矩阵
for i in range(1, N - 1):
    A_matrix[i, i - 1] = 1  # 左边节点
    A_matrix[i, i] = -2     # 当前节点
    A_matrix[i, i + 1] = 1  # 右边节点

# 设置边界条件
A_matrix[0, 0] = -2
A_matrix[0, 1] = 1
A_matrix[-1, -1] = -2
A_matrix[-1, -2] = 1

# 设置右端项
b[0] = -T_A
b[-1] = -T_B

# 求解线性方程组 A_matrix * T = b
T = np.linalg.solve(A_matrix, b)

# 在左边和右边插入边界温度
T_full = np.concatenate(([T_A], T, [T_B]))

# 输出结果
print("Temperature distribution:", T_full)

# 定义位置
x = np.linspace(0, L, N + 2)  # 包括边界的 N+2 个点

# 绘制温度分布
plt.plot(x, T_full, marker='o')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution in 1D Steady-State Diffusion')
plt.grid(True)
plt.show()
