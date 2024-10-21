import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

# Parameters for the problem
L = 1.0 # Length of the rod
T = 0.5 # Total time
alpha = 0.01 # Thermal diffusivity
Nx = 10 # Increased number of elements (nodes are Nx+1)
Nt = 500 # Number of time steps
dx = L / Nx # Spatial step size
dt = T / Nt # Time step size

# Generate the grid points (nodes)
x = np.linspace(0, L, Nx + 1)

# Initial condition: u(x,0) = sin(pi * x)
u = np.sin(np.pi * x)
u_new = np.zeros_like(u) # Array to store new temperature values

# Boundary conditions
u[0] = 0 # u(0, t) = 0
u[-1] = 0 # u(L, t) = 0

# Stiffness matrix K (tridiagonal)
main_diag_K = 2 / dx * np.ones(Nx - 1)
off_diag_K = -1 / dx * np.ones(Nx - 2)

# Mass matrix M (tridiagonal)
main_diag_M = 2 * dx / 6 * np.ones(Nx - 1)
off_diag_M = dx / 6 * np.ones(Nx - 2)

# Use scipy.sparse.diags to create the tridiagonal matrices
K = diags([off_diag_K, main_diag_K, off_diag_K], [-1, 0, 1])
M = diags([off_diag_M, main_diag_M, off_diag_M], [-1, 0, 1])

# Convert to CSR format to be compatible with spsolve
K = csc_matrix(K)
M = csc_matrix(M)

# Time-stepping loop
for n in range(Nt):
    # Right-hand side: M * u
    rhs = M @ u[1:Nx] # Interior nodes only (excluding boundary)

    # System matrix: A = M + alpha * dt * K
    A = M + alpha * dt * K

    # Solve for the next time step: (M + alpha * dt * K) u_new = M * u
    u_new[1:Nx] = spsolve(A, rhs)

    # Update boundary conditions explicitly (keep them zero)
    u_new[0] = 0
    u_new[-1] = 0

    # Update for the next time step
    u[:] = u_new[:]

# Plot the final solution
plt.plot(x, u, label="Numerical solution (FE)")
plt.plot(x, np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * x), 'r--', label="Analytical solution")
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('1D Heat Equation - Finite Element Method')
plt.legend()
plt.grid(True)
plt.show()