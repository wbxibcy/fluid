import numpy as np
import matplotlib.pyplot as plt

# 参数设置
L = 1.0              # 腔体边长
Nx, Ny = 50, 50      # 网格数量
dx = L / Nx          # 网格尺寸
dy = L / Ny
Re = 100             # 雷诺数
U = 1.0              # 顶部速度
nu = U * L / Re      # 黏性系数
rho = 1.0            # 流体密度
dt = 0.01            # 时间步长
nt = 500             # 时间步数

# 初始化变量
u = np.zeros((Nx+2, Ny+2))  # 水平速度
v = np.zeros((Nx+2, Ny+2))  # 垂直速度
p = np.zeros((Nx+2, Ny+2))  # 压力场

# 边界条件函数
def set_boundary_conditions(u, v):
    u[:, -1] = U     # 顶部边界：u = U
    u[:, 0] = 0      # 底部边界：u = 0
    u[0, :] = 0      # 左边界：u = 0
    u[-1, :] = 0     # 右边界：u = 0
    
    v[:, -1] = 0     # 顶部边界：v = 0
    v[:, 0] = 0      # 底部边界：v = 0
    v[0, :] = 0      # 左边界：v = 0
    v[-1, :] = 0     # 右边界：v = 0

# 离散化方程
def solve_momentum(u, v, p):
    un = u.copy()
    vn = v.copy()

    # 预测 u
    u[1:-1, 1:-1] = (
        un[1:-1, 1:-1]
        - dt / dx * (un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]))
        - dt / dy * (vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]))
        + nu * dt * (
            (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dx**2
            + (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dy**2
        )
        - dt / (2 * rho * dx) * (p[2:, 1:-1] - p[:-2, 1:-1])
    )

    # 预测 v
    v[1:-1, 1:-1] = (
        vn[1:-1, 1:-1]
        - dt / dx * (un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]))
        - dt / dy * (vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]))
        + nu * dt * (
            (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dx**2
            + (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dy**2
        )
        - dt / (2 * rho * dy) * (p[1:-1, 2:] - p[1:-1, :-2])
    )
    return u, v

def solve_pressure(u, v, p):
    pn = p.copy()
    p[1:-1, 1:-1] = (
        ((pn[2:, 1:-1] + pn[:-2, 1:-1]) * dy**2
        + (pn[1:-1, 2:] + pn[1:-1, :-2]) * dx**2
        - rho * dx**2 * dy**2 / dt * (
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
            + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        )) / (2 * (dx**2 + dy**2))
    )
    return p

# 主循环
for n in range(nt):
    set_boundary_conditions(u, v)
    u, v = solve_momentum(u, v, p)
    p = solve_pressure(u, v, p)

# 可视化结果
plt.contourf(u[1:-1, 1:-1], cmap="jet")
plt.colorbar()
plt.title("Velocity Field (u) at Re=100")
plt.show()
