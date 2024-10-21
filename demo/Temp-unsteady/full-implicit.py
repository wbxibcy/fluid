import numpy as np
import matplotlib.pyplot as plt

# 板的物理参数
L = 0.02  # 板的厚度，单位：m
k = 10  # 热导率，单位：W/m/K
rho_c = 10e6  # 体积热容，单位：J/m³/K
initial_temp = 200  # 初始温度，单位：摄氏度
boundary_temp = 0  # 边界温度，单位：摄氏度
dx = 0.004  # 空间步长，单位：m

# 离散化
nx = int(L / dx) + 1  # 空间离散后的节点数
x = np.linspace(0, L, nx)

# 时间步长为8秒
dt = 8  # 时间步长，单位：秒

# 时间参数
t_end = 20  # 只计算到40秒
n_steps = int(t_end / dt)  # 时间步数

# 初始化温度分布
T = np.full(nx, initial_temp)

def build_matrix(nx, alpha):
    A = np.zeros((nx, nx))
    
    # 左边界：增加半个格子的影响
    A[0, 0] = 1 - alpha  # 半个格子体积贡献到左边界点
    A[0, 1] = -alpha  # 连接左边界与下一个节点

    # 内部节点的三对角矩阵部分
    for i in range(1, nx - 1):
        A[i, i - 1] = -alpha  # 西侧
        A[i, i] = 1 + 2 * alpha  # 当前节点
        A[i, i + 1] = -alpha  # 东侧

    # 右边界：增加半个格子的影响
    A[-1, -2] = -alpha  # 连接右边界与倒数第二个节点
    A[-1, -1] = 1 + 3 * alpha  # 半个格子体积贡献到右边界点

    print(A)
    return A

# 全隐式求解法
def full_implicit_fvm(T, nx, dt, dx, k, rho_c, boundary_temp, n_steps):
    alpha = k * dt / (rho_c * dx**2)
    A = build_matrix(nx, alpha)  # 构造三对角矩阵
    b = T.copy()  # 初始化b向量

    for _ in range(n_steps):
        b[0] = boundary_temp  # 绝热边界，T[0] = T[1]
        b[-1] = boundary_temp  # 东侧温度为0°C
        T = np.linalg.solve(A, b)  # 解三对角矩阵方程
    return T

# 解析解（Ozisik）
def analytical_solution_ozisik(x, t, L, k, rho_c, initial_temp, num_terms=1000):
    """基于Ozisik给出的解析解"""
    alpha = k / rho_c  # 热扩散率
    T_analytical = np.zeros_like(x)  # 初始化温度数组
    for n in range(1, num_terms + 1):
        # 计算每一项的 lambda_n
        lambda_n = (2 * n - 1) * np.pi / (2 * L)
        # 计算傅里叶级数中的每一项
        term = ((-1)**(n + 1) / (2 * n - 1)) * \
               np.exp(-alpha * lambda_n**2 * t) * \
               np.cos(lambda_n * x)
        T_analytical += term
    # 将傅里叶级数的结果乘以系数
    T_analytical = (4 * initial_temp / np.pi) * T_analytical
    return T_analytical

# 使用全隐式格式计算温度分布
T_implicit = full_implicit_fvm(T, nx, dt, dx, k, rho_c, boundary_temp, n_steps)

# 在 t=40秒 时进行解析解对比
T_ana = analytical_solution_ozisik(x, t_end, L, k, rho_c, initial_temp)

# 绘图
plt.plot(x, T_implicit, label=f'Fully Implicit t={t_end:.0f}s')
plt.plot(x, T_ana, '--', label=f'Analytical (Ozisik) t={t_end:.0f}s')

plt.xlabel('Position along the slab (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title(f'Temperature Distribution at t={t_end} seconds using Fully Implicit vs Analytical')
plt.show()
