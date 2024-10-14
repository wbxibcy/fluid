import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cavity_flow(u, v, p, nx, ny, save_path='result', filename='cavity_flow.png'):
    # 创建保存文件夹
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.figure(figsize=(11, 7), dpi=100)
    plt.contourf(np.linspace(0, 2, nx), np.linspace(0, 2, ny), p, alpha=0.5, cmap='viridis')
    plt.colorbar()
    plt.contour(np.linspace(0, 2, nx), np.linspace(0, 2, ny), p, cmap='viridis')
    plt.quiver(np.linspace(0, 2, nx), np.linspace(0, 2, ny), u, v)
    plt.xlabel('X')
    plt.ylabel('Y')

    fig = plt.gcf()
    plt.show()

    # 保存图片
    save_file = os.path.join(save_path, filename)
    fig.savefig(save_file, dpi=200)
    plt.close()

    print(f"图片已保存到: {save_file}")

