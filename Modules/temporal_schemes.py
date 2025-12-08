from numpy.linalg import norm, solve
from numpy import zeros
from .math_functions import Newton





def Euler(U, t1, t2, F):
    dt = t2 - t1 
    return U + dt * F(U, t1)

def Inverse_Euler(U, t1, t2, F):
    dt = t2 - t1
    def G(x):
        return x - U - dt * F(x, t2)
    return Newton(G, U)

def Crank_Nicolson(U, t1, t2, F):
    dt = t2 - t1 
    a = U + dt / 2 * F(U, t1)
    def G(x):
        return x - a - dt/2 * F(x, t2)
    return Newton(G, U)

def RK4(U, t1, t2, F):
    dt = t2 - t1
    k1 = F(U, t1)
    k2 = F(U + 0.5 * dt * k1, t1 + 0.5 * dt)
    k3 = F(U + 0.5 * dt * k2, t1 + 0.5 * dt)
    k4 = F(U + dt * k3, t2)
    return U + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def Cauchy(F, U0, t, esquema):
    N = len(t) - 1
    U = zeros((N + 1, len(U0)))
    U[0, :] = U0

    for n in range(N):     
        U[n+1, :] = esquema(U[n, :], t[n], t[n+1], F)

    return U

def Leap_Frog(x0, v0, t0, tf, dt):
    N = int((tf - t0) / dt)

    t = zeros(N + 1)
    x = zeros(N + 1)
    v = zeros(N + 1)

    t[0] = t0
    x[0] = x0
    v[0] = v0

    for n in range(N):
        t[n+1] = t[n] + dt
        a_n = -x[n]
        v_half = v[n] + 0.5 * dt * a_n
        x[n+1] = x[n] + dt * v_half
        a_np1 = -x[n+1]
        v[n+1] = v_half + 0.5 * dt * a_np1

    return x, v