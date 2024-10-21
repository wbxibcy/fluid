import numpy as np
import matplotlib.pyplot as plt

def chebyshev(N):
    """
    Function to compute the Chebyshev nodes and differentiation matrix
    """
    x = np.cos(np.pi * np.arange(N + 1) / N) # Chebyshev-Gauss-Lobatto nodes in [-1, 1]
    c = np.ones(N + 1)
    c[0] = 2
    c[-1] = 2
    c[1:N] = 1
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = (c * (1 / c).T) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

# Parameters
alpha = 0.01 # Thermal diffusivity
T = 0.5 # Total time
Nt = 500 # Number of time steps
N = 40 # Number of Chebyshev nodes (polynomial degree)
dt = T / Nt # Time step size

# Get Chebyshev differentiation matrix and nodes
D, x_cheb = chebyshev(N)

# Map the Chebyshev nodes from [-1, 1] to [0, 1]
x = (x_cheb + 1) / 2

# Initial condition: u(x,0) = sin(pi * x) on [0, 1]
u = np.sin(np.pi * x)

# Apply boundary conditions (u(0) = u(1) = 0)
u[0] = 0
u[-1] = 0

# Crank-Nicolson scheme for time-stepping
I = np.eye(N + 1)
A = I - 0.5 * alpha * dt * D @ u # Crank-Nicolson system matrix
B = I + 0.5 * alpha * dt * D @ u

# Time-stepping loop
for n in range(Nt):
    # Update u using Crank-Nicolson (solve A * u_new = B * u_old)
    u_new = np.linalg.solve(A, B @ u)
    u_new[0] = 0 # Enforce boundary condition at x=0
    u_new[-1] = 0 # Enforce boundary condition at x=1
    u = u_new

# Exact solution for comparison
u_exact = np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * x)

# Plot the results
plt.plot(x, u, label="Numerical solution (Chebyshev)")
plt.plot(x, u_exact, 'r--', label="Analytical solution")
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('1D Heat Equation - Chebyshev Spectral Method')
plt.legend()
plt.grid(True)
plt.show()