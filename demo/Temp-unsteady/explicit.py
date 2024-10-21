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
t_end = 40  # 只计算到40秒
n_steps = int(t_end / dt)  # 时间步数

# 初始化温度分布
T = np.full(nx, initial_temp)
T_new = T.copy()

# 显式有限体积法
def explicit_fvm(T, nx, dt, dx, k, rho_c, boundary_temp):
    T_new = T.copy()
    alpha = k * dt / (rho_c * dx**2)
    for i in range(1, nx - 1):
        T_new[i] = T[i] + alpha * (T[i-1] - 2 * T[i] + T[i+1])
    # 边界条件：东侧为0°C，西侧绝热
    T_new[0] = T_new[1]  # 绝热边界，温度梯度为0
    T_new[-1] = boundary_temp  # 东侧温度为0°C
    return T_new

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

# 主循环，步长为8秒，计算到t=40秒
for step in range(n_steps):
    T = explicit_fvm(T, nx, dt, dx, k, rho_c, boundary_temp)

# 在 t=40秒 时进行Özışık解析解对比
T_ana = analytical_solution_ozisik(x, t_end, L, k, rho_c, initial_temp)

# 绘图
plt.plot(x, T, label=f'Numerical t={t_end:.0f}s')
plt.plot(x, T_ana, '--', label=f'Analytical (Ozisik) t={t_end:.0f}s')

plt.xlabel('Position along the slab (m)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title(f'Temperature Distribution at t={t_end} seconds')
plt.show()
