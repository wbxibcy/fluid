import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch
if torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
    print("Using GPU")
else:
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    print("Using CPU")

# 定义问题的几何区域
geom = dde.geometry.Rectangle([0, 0], [1, 1])  # 空间范围 x ∈ [0, 1], y ∈ [0, 1]

# 雷诺数
Re = 100
nu = 1 / Re  # 粘性系数

# 定义 PDE
def pde(x, y):
    """
    Navier-Stokes 方程组：
    - Momentum equations (u, v)
    - Continuity equation (incompressible flow)
    """
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # 一阶导数
    u_x = dde.grad.jacobian(y, x, i=0, j=0)  # ∂u/∂x
    u_y = dde.grad.jacobian(y, x, i=0, j=1)  # ∂u/∂y
    v_x = dde.grad.jacobian(y, x, i=1, j=0)  # ∂v/∂x
    v_y = dde.grad.jacobian(y, x, i=1, j=1)  # ∂v/∂y
    p_x = dde.grad.jacobian(y, x, i=2, j=0)  # ∂p/∂x
    p_y = dde.grad.jacobian(y, x, i=2, j=1)  # ∂p/∂y

    # 二阶导数
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    # 动量方程
    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    # 连续性方程
    continuity = u_x + v_y

    return [momentum_x, momentum_y, continuity]

# 静止边界 (u = v = 0)
def boundary_static(x, on_boundary):
    """静止边界条件: 包括左、右、底边"""
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[0], 0) or np.isclose(x[0], 1))

def boundary_top(x, on_boundary):
    """顶盖移动的边界条件: u = 1, v = 0"""
    return on_boundary and np.isclose(x[1], 1)

# 边界条件表达式
bc1 = dde.DirichletBC(geom, lambda x: 0, boundary_static, component=0)  # u = 0
bc2 = dde.DirichletBC(geom, lambda x: 0, boundary_static, component=1)  # v = 0
bc3 = dde.DirichletBC(geom, lambda x: 1, boundary_top, component=0)          # u = 1
bc4 = dde.DirichletBC(geom, lambda x: 0, boundary_top, component=1)          # v = 0

# 设置数据
data = dde.data.PDE(geom, pde, [bc1, bc2, bc3, bc4], num_domain=5000, num_boundary=500)

# 神经网络结构
net = dde.nn.FNN([2] + [100] * 8 + [3], "swish", "Glorot uniform")

# 模型
model = dde.Model(data, net)

# L-BFGS 优化器
model.compile("L-BFGS")
losshistory, train_state = model.train(iterations=10000)

# 预测流场
X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
u, v, p = model.predict(xy).T

# 将预测的结果重塑为网格
u = u.reshape(100, 100)
v = v.reshape(100, 100)
p = p.reshape(100, 100)

# 计算速度大小
velocity_magnitude = np.sqrt(u**2 + v**2)

# 绘制热力图和流线图
plt.figure(figsize=(12, 8))
plt.contourf(X, Y, velocity_magnitude, levels=50, cmap="jet")  # 热力图
plt.colorbar(label="Velocity Magnitude")
plt.streamplot(X, Y, u, v, color="k", density=1.2)  # 流线图
plt.title("Velocity Field Heatmap and Streamlines (Re = 100)")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("Velocity Field Heatmap and Streamlines (Re = 100).png", dpi=300)
plt.show()
