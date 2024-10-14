import numpy as np
import matplotlib.pyplot as plt

# 参数
L = 0.5  # 杆的长度 (m)
T_A = 100  # 左端的温度 (°C)
T_B = 500  # 右端的温度 (°C)
k = 1000  # 导热系数 (W/mK)
N = 10  # 网格点数（内部节点数）
dx = L / (N + 1)  # 网格间距

# 系数矩阵 A 和右侧向量 b
A = np.zeros((N, N))
b = np.zeros(N)

# 填充系数矩阵 A
for i in range(1, N - 1):
    A[i, i-1] = 1  # T_{i-1}
    A[i, i] = -2  # T_i
    A[i, i+1] = 1  # T_{i+1}

# 边界条件的影响
A[0, 0] = -2
A[0, 1] = 1
A[N-1, N-2] = 1
A[N-1, N-1] = -2

# 右侧向量 b
b[0] = -T_A
b[-1] = -T_B

# 解决线性方程组
T_internal = np.linalg.solve(A, b)

# 加入边界条件
T_full = np.concatenate(([T_A], T_internal, [T_B]))

# 输出结果
print("Temperature distribution:", T_full)

# 定义位置
x = np.linspace(0, L, N + 2)  # 包括边界的 N+2 个点

# 绘制温度分布
plt.plot(x, T_full, marker='o')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution using Finite Difference Method')
plt.grid(True)
plt.show()
