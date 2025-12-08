from numpy import array, sqrt, linspace, zeros
import matplotlib.pyplot as plt
from scipy.optimize import newton

t0 = 0
tf = 7
N = 2000

dt = (tf-t0)/N
t = linspace(t0, tf, N+1)
U0 = array([1,0,0,1]) 


# KEPLER
def kepler (U, t):
   x = U [0]
   y = U [1]
   vx = U [2]
   vy = U [3]

   r = sqrt(x**2 + y**2)

   F = (vx, vy, - (x / r**3), - (y / r**3))

   return array (F)

# 1) EULER CON CAUCHY
def euler (U, dt, t, F):
    
    return U + dt * F(U,t)  


def cauchy (U0, t, F, esquema):
    
    U = zeros([N+1, len(U0)])
    U[0, :] = U0
    
    for n in range(0, N): 
        U[n+1,:] = esquema( U[n, :], t[n+1] - t[n], t[n],  F )
    
    return U


# 2) CRANK NICOLSON
def crank_nicolson(U, dt, t, F):

    def G(X): 

        return X - U - (dt/2) *(F(U,t) + F(X,t))
    
    return newton(G, U)


# 3) RUNGE KUTTA 4
def RK4 (U, dt, t, F):

    k1 = F (U, t)
    k2 = F ( U + k1 * dt/2, t + dt/2)
    k3 = F ( U + k2 * dt/2, t + dt/2)
    k4 = F ( U + k3 * dt , t + dt/2)

    return U + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)


# 4) EULER INVERSO
def inverse_euler (U, dt, t, F):
       
    def G(X):
        return X - U - dt*F(X,t)

    return newton(G, U)
   

# 5) INTEGRAR CAUCHY

# 5.1) EULER
U_euler = cauchy (U0, t, kepler , euler)

# 5.2) CRANK NICOLSON
U_CN = cauchy (U0, t, kepler, crank_nicolson )

# 5.3) RK4
U_RK4 = cauchy (U0, t, kepler, RK4)

# 5.4) EULER INVERSO
U_inverse_euler = cauchy (U0, t, kepler, inverse_euler)

# GRÁFICAS
plt.figure()
plt.plot(U_euler[:, 0], U_euler[:, 1], label="Euler Explícito")
plt.plot(U_RK4[:, 0], U_RK4[:, 1], label="Runge-Kutta 4 etapas")
plt.plot(U_CN[:, 0], U_CN[:, 1], label="Crank-Nickolson")
plt.plot(U_inverse_euler[:,0], U_inverse_euler[:,1], label="Euler Inverso", alpha=0.6)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Resolución EDO")
plt.grid()
plt.show()