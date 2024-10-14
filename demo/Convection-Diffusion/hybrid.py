# u=0.1m/s，网格节点数为 5
import numpy as np
import matplotlib.pyplot as plt

# 解析解函数
def analytical_solution(u, rho, Gamma, L, x):
    phi_0 = 1.0  # 边界条件 x = 0
    phi_L = 0.0  # 边界条件 x = L
    return phi_0 + (phi_L - phi_0) * (np.exp(rho * u * x / Gamma) - 1) / (np.exp(rho * u * L / Gamma) - 1)

# 有限体积法（Hybrid格式）
def finite_volume_method_hybrid(u, rho, Gamma, L, nx):
    dx = L / (nx - 1)  # 网格间距
    F = rho * u  # 对流通量
    phi = np.zeros(nx)
    phi[0] = 1.0  # 边界条件 x = 0
    A = np.zeros((nx, nx))
    b = np.zeros(nx)

    # 设置系数矩阵A和右端项b
    for i in range(1, nx - 1):
        Pe = rho * u * dx / Gamma  # Peclet 数

        if abs(Pe) < 2:
            # 中心差分格式
            a_W = Gamma / dx + F / 2  # 西端系数
            a_E = Gamma / dx - F / 2  # 东端系数
        else:
            # 迎风格式
            a_W = Gamma / dx + F  # 西端系数
            a_E = Gamma / dx   # 东端系数

        a_P = a_W + a_E  # 当前点系数

        A[i, i - 1] = -a_W  # 西端
        A[i, i] = a_P       # 当前点
        A[i, i + 1] = -a_E  # 东端

    A[0, 0] = 1.0  # 边界条件 x=0
    A[-1, -1] = 1.0  # 边界条件 x=L
    b[0] = 1.0
    b[-1] = 0.0  # 边界条件 x=L

    # 解线性方程
    phi = np.linalg.solve(A, b)
    return phi

def plot_results(x, phi_fd, phi_analytical):
    plt.plot(x, phi_fd, 'bo-', label='FVM (Hybrid) Solution')
    plt.plot(x, phi_analytical, 'r--', label='Analytical Solution')
    plt.xlabel('x (m)')
    plt.ylabel('phi')
    plt.title('Case 1: u = 0.1 m/s, FVM Hybrid vs Analytical Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 参数设置
    L = 1.0  # 长度
    rho = 1.0  # 密度
    Gamma = 0.1  # 扩散系数
    u = 0.1  # 对流速度
    nx = 5  # 网格点数
    x = np.linspace(0, L, nx)

    # 有限体积法求解
    phi_fd = finite_volume_method_hybrid(u, rho, Gamma, L, nx)

    # 解析解
    phi_analytical = analytical_solution(u, rho, Gamma, L, x)

    # 绘图
    plot_results(x, phi_fd, phi_analytical)

if __name__ == '__main__':
    main()
