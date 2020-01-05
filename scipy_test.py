"""Comparison of scipy's integration functions.
"""
import time
from functools import partial
import numpy as np
import scipy
from scipy.integrate import odeint, solve_ivp


def lorenz_odes(t, y, sigma, beta, rho):
    """The Lorenz system of ordinary differential equations.

    Returns:
        dydt (tuple): Derivative (w.r.t. time)
    """
    y1, y2, y3 = y
    return (sigma * (y2 - y1), y1 * (rho - y3) - y2, y1 * y2 - beta * y3)


print(f"Scipy: {scipy.__version__}")

dt = 0.01
T = 50
t = np.arange(dt, T + dt, dt)

# Lorenz system parameters
beta = 8 / 3
sigma = 10
rho = 28
n = 3

# Function to be integrated - with parameter values
fun = partial(lorenz_odes, sigma=sigma, beta=beta, rho=rho)

# Initial condition
y0 = (-8, 8, 27)

# Simulate using scipy.integrate.odeint method
# Produces same results as Matlab
rtol = 10e-12
atol = 10e-12 * np.ones_like(y0)
t0 = time.time()
y = odeint(fun, y0, t, tfirst=True, rtol=rtol, atol=atol)
print(f"odeint: {(time.time() - t0)*1000:.1f} ms")
assert y.shape == (5000, 3)

# Simulate using scipy.integrate.solve_ivp method
t_span = [t[0], t[-1]]
t0 = time.time()
sol = solve_ivp(fun, t_span, y0, t_eval=t, method='LSODA')
print(f"solve_ivp(LSODA): {(time.time() - t0)*1000:.1f} ms")

assert sol.t.shape == t.shape
assert sol.y.swapaxes(0, 1).shape == y.shape

print(f"Max difference: {np.max(y - sol.y.swapaxes(0, 1)):.3f}")
