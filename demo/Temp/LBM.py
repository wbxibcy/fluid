import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 0.5  # 杆的长度 (m)
T_A = 100  # 左边界温度 (°C)
T_B = 500  # 右边界温度 (°C)

# 离散化参数
N = 100  # 网格数量
dx = L / N  # 网格长度

# 初始化温度场
T = np.zeros(N)  # 温度场
T[0] = T_A  # 左边界温度
T[-1] = T_B  # 右边界温度

# LBM参数
tau = 0.5  # 放松时间，设置为较小值
max_iterations = 10000  # 最大迭代次数
tolerance = 1e-5  # 收敛容忍度

# 主迭代过程
for it in range(max_iterations):
    T_old = T.copy()  # 复制当前温度场
    
    # 计算新温度场，采用简单的热传导差分法
    for i in range(1, N - 1):
        T[i] = (T_old[i - 1] + T_old[i + 1]) / 2  # 平均值更新

    # 收敛检查
    if np.max(np.abs(T - T_old)) < tolerance:
        print(f'Converged after {it} iterations.')
        break

# 输出结果
print("Temperature distribution:", T)

# 定义位置
x = np.linspace(0, L, N)

# 绘制温度分布
plt.plot(x, T, marker='o')
plt.xlabel('Position (m)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution in 1D Steady-State Diffusion using LBM')
plt.grid(True)
plt.show()
