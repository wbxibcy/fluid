import numpy as np
import matplotlib.pyplot as plt

# Parameters for the problem
L = 1.0 # Length of the rod
T = 0.5 # Total time
alpha = 0.01 # Thermal diffusivity
Nx = 64 # Number of spatial grid points (this also limits the number of Fourier modes)
Nt = 500 # Number of time steps
dt = T / Nt # Time step size
N = 2 # Number of Fourier modes to use

# Generate the spatial grid points
x = np.linspace(0, L, Nx, endpoint=False)

# Initial condition: u(x,0) = sin(pi * x)
u = np.sin(np.pi * x)

# Fourier transform of the initial condition
u_hat = np.fft.rfft(u)

# Wavenumbers for the spectral method (only up to N modes)
k = np.fft.rfftfreq(Nx, d=(x[1] - x[0])) * 2 * np.pi
k = k[:N+1] # Take only the first N modes (including the zeroth mode)

# Keep only the first N Fourier modes, discard higher modes
u_hat = u_hat[:N+1]

# Time-stepping loop
for n in range(Nt):
 # Update the Fourier coefficients for the first N modes using the analytical time evolution
 u_hat *= np.exp(-alpha * (k**2) * dt)
# Transform back to physical space using only the first N Fourier modes
u_final = np.fft.irfft(u_hat, n=Nx)

# Plot the final solution
plt.plot(x, u_final, label=f"Numerical solution (Fourier)")
plt.plot(x, np.exp(-alpha * np.pi**2 * T) * np.sin(np.pi * x), 'r--', label="Analytical solution")
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title(f'1D Heat Equation - Fourier Spectral Method')
plt.legend()
plt.grid(True)
plt.show()