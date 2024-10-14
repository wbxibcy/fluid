from solver import cavity_flow
from visualization import plot_cavity_flow

import numpy as np

# 参数设置
nx, ny = 41, 41  # 网格点数
nt = 500  # 时间步数
nit = 50  # 迭代次数（用于解压力泊松方程）
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
dt = 0.001  # 时间步长

# 物理参数
rho = 1  # 流体密度
nu = 0.1  # 流体动力粘度

# 初始化速度和压力场
u = np.zeros((ny, nx))  # x方向速度
v = np.zeros((ny, nx))  # y方向速度
p = np.zeros((ny, nx))  # 压力

# 运行模拟
u, v, p = cavity_flow(nx, ny, nt, nit, u, v, dt, dx, dy, p, rho, nu)

# 可视化并保存图片
plot_cavity_flow(u, v, p, nx, ny, save_path='result', filename='cavity_flow.png')