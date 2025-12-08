from numpy.linalg import norm, solve
from numpy import zeros, concatenate


def F(U, t):
    x = U[:2]
    vel = U[2:]
    x_norm = norm(x)
    acc = -x / (x_norm**3)
    return concatenate((vel, acc))
