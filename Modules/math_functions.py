from numpy.linalg import norm, solve
from numpy import zeros, concatenate


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