from numpy.linalg import norm, solve
from numpy import zeros, concatenate

def F(U, t):
    x = U[:2]
    vel = U[2:]
    x_norm = norm(x)
    acc = -x / (x_norm**3)
    return concatenate((vel, acc))

def Derivada(f, x, dx, h = 1e-7):
    return (f(x + dx) - f(x - dx)) / (2 * h)

def Jacobiano(f, x, h = 1e-7):
    n = len (x)
    J = zeros((n, n))
    for j in range(n):
        dx = zeros(n)
        dx[j] = h
        J[:, j] = Derivada(f, x, dx)
    return J

def Gauss(A, b):
    return solve(A, b)

def Newton(f, x0, tol=1e-8, max_iter=30):
    x = x0.copy()
    Dx = 1.0

    for k in range(max_iter):
        A = Jacobiano(f, x)
        Dx = Gauss(A, -f(x))
        x = x + Dx

        if norm(Dx) < tol:
            return x
    return x


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

