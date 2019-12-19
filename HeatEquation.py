import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags


class HeatEquation():
    """Finite Differences for the Heat Equation PDE"""
    
    def __init__(self, f, L=1, T=1, dx=0.1, dt=0.00025):
        """Initialize the PDE.
        
        Parameters
        ----------
        f : function
            The callable for the boundary condition
        L : float
            The rod length
        T : float
            Stopping time
        dx : float
            Step length in x-dimension
        dt : float
            Step length in t-dimension
        
        """
        if dt <= 0 or dx <= 0:
            raise ValueError("dt and dx must be greater than 0")
        if L <= 0 or T <= 0:
            raise ValueError("L and T must be greater than 0")

        self.T = T   # Stopping time
        self.L = L   # Rod length
        self.dx = dx # x step
        self.dt = dt # Time step
        self.f = f   # Initial conditions, callable
        self.alpha = round(dt/dx**2, 5)

        self.n_t = int(self.T/self.dt) + 1
        self.n_x = int(self.L/self.dx) + 1
        
        self.x = np.zeros(self.n_x)
        self.t = np.zeros(self.n_t)
        self.g = np.zeros((self.n_x, self.n_t))
        self.t[:-1] = np.arange(0, self.T, self.dt)
        self.t[-1] = self.T
    
        self.x[:-1] = np.arange(0, self.L, self.dx)
        self.x[-1] = self.L
    
        self.g[0, :] = 0.0 # self.u = a(t), boundary condition u(0, t)
        self.g[-1, :] = 0.0 # self.u = b(t), boundary condition u(L, t)
        self.g[1:-1, 0] = self.f(self.x[1:-1]) # self.u = b(t), initial condition u(x, 0)
        self.IsSolved = False
    
    def __call__(self, j):
        """Return the x-space vector at j in time."""
        return self.g[:, j]

    def ExplicitEuler(self):
        """Solve PDE with explicit Euler method."""
        a1 = 1 - 2 * self.alpha
        a2 = self.alpha

        # Set up sparse A-matrix
        self.A = diags([a2, a1, a2], 
                       [-1, 0,  1], 
                       shape=(self.n_x - 2, self.n_x - 2))

        for i in range(self.n_t - 1):
            self.g[1:-1, i + 1] = self.A @ self.g[1:-1, i]

        self.IsSolved = True


