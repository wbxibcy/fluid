import numpy as np
import matplotlib.pyplot as plt

# Parameters for the problem
L = 1.0 # Length of the rod
T = 0.5 # Total time
alpha = 0.01 # Thermal diffusivity
Nx = 10 # Number of spatial points
Nt = 500 # Number of time steps
dx = L / (Nx - 1) # Spatial step size
dt = T / Nt # Time step size
r = alpha * dt / dx**2 # Stability condition parameter

# Create grid
x = np.linspace(0, L, Nx)
u = np.sin(np.pi * x) # Initial condition: u(x,0) = sin(pi*x)
u_new = np.zeros_like(u) # Array to store new temperature values

# Boundary conditions
u[0] = 0 # u(0, t) = 0
u[-1] = 0 # u(L, t) = 0

# Time-stepping loop
for n in range(Nt):
    # Update the temperature using the finite difference formula
    for i in range(1, Nx-1):
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])

    # Update for the next time step
    u[:] = u_new[:]

# Plot the final solution
plt.plot(x, u, label="Numerical solution (FD)")
plt.plot(x, np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * x), 'r--', label="Analytical solution")
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('1D Heat Equation - Finite Difference Method')
plt.legend()
plt.grid(True)
plt.show()