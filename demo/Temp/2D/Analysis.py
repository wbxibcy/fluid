# Dirichlet 边界条件
# 分离变量法

import numpy as np
import matplotlib.pyplot as plt

# 定义区域尺寸
Lx = 0.5
Ly = 1.0

# 创建网格
nx = 100
ny = 100
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# 初始化温度场
T = np.zeros((ny, nx))

# 解析解计算
for i in range(nx):
    for j in range(ny):
        T[j, i] = (800 * x[i] + 100) * (250 * y[j] + 50)

# 绘制结果
plt.figure(figsize=(8, 4))
plt.contourf(x, y, T, levels=50, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.title('Steady-State Temperature Distribution (Analytical Solution)')
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.show()

