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

# 定义解析解函数
def analytical_solution(u, rho, Gamma, L, x):
    phi_0 = 1.0  # 边界条件 x = 0
    phi_L = 0.0  # 边界条件 x = L
    return phi_0 + (phi_L - phi_0) * (np.exp(rho * u * x / Gamma) - 1) / (np.exp(rho * u * L / Gamma) - 1)

# 定义稳态对流扩散方程的PDE
def pde(x, phi):
    u = 0.1  # 对流速度
    rho = 1.0  # 密度
    Gamma = 0.1  # 扩散系数
    dphi_dx = dde.grad.jacobian(phi, x, i=0)
    d2phi_dx2 = dde.grad.hessian(phi, x, i=0)
    return u * rho * dphi_dx - Gamma * d2phi_dx2

# 边界条件
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

# 边界值
def func_boundary_left(x):
    return 1.0

def func_boundary_right(x):
    return 0.0

# 主函数
def main():
    L = 1.0  # 长度
    rho = 1.0  # 密度
    Gamma = 0.1  # 扩散系数
    u = 0.1  # 对流速度
    nx = 100  # 网格点数
    x = np.linspace(0, L, nx)

    # 定义DeepXDE问题
    geom = dde.geometry.Interval(0, L)
    bc_left = dde.DirichletBC(geom, func_boundary_left, boundary_left)
    bc_right = dde.DirichletBC(geom, func_boundary_right, boundary_right)

    # 定义问题
    data = dde.data.PDE(
        geom,
        pde,
        [bc_left, bc_right],
        num_domain=20,
        num_boundary=2,
    )

    # 定义神经网络
    net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # 训练模型
    model.compile("adam", lr=1e-3)
    losshistory, train_state = model.train(iterations=1000)

    # 获取预测值
    x_test = np.linspace(0, L, nx).reshape(-1, 1)
    phi_pred = model.predict(x_test)

    # 解析解
    phi_analytical = analytical_solution(u, rho, Gamma, L, x)

    # 绘图
    plt.plot(x, phi_analytical, "r--", label="Analytical Solution")
    plt.plot(x_test, phi_pred, "b.-", label="DeepXDE Prediction")
    plt.xlabel("x (m)")
    plt.ylabel("phi")
    plt.title("Steady Convection-Diffusion: DeepXDE vs Analytical")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
