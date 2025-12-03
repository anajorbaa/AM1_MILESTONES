from numpy import sqrt, zeros, linspace, array
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


### 1) Write a high order embedded Runge-Kutta method

# FEHLBERG: 6 ETAPAS
def RK_Fehlberg(U, t1, t2, F):
    h = t2 - t1

    k1 = F(t1, U)
    k2 = F(t1 + h/4, U + h*(1/4)*k1)
    k3 = F(t1 + 3*h/8, U + h*(3/32*k1 + 9/32*k2))
    k4 = F(t1 + 12*h/13, U + h*(1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3))
    k5 = F(t1 + h, U + h*(439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4))
    k6 = F(t1 + h/2, U + h*(-8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5))

    U5 = U + h*(16/135*k1 + 6656/12825*k2 + 28561/56430*k3 + 9/50*k4 + 2/55*k5)
    U4 = U + h*(25/216*k1 + 1408/2565*k2 + 2197/4104*k3 - 1/5*k4)

    return U5


# DORMAND - PRINCE: 7 ETAPAS 
def RK_DormandPrince(U, t1, t2, F):
    h = t2 - t1

    k1 = F(t1, U)
    k2 = F(t1 + h*(1/5), U + h*(1/5)*k1)
    k3 = F(t1 + h*(3/10), U + h*(3/40)*k1 + (9/40)*k2)
    k4 = F(t1 + h*(4/5), U + h*(44/45)*k1 - (56/15)*k2 + (32/9)*k3)
    k5 = F(t1 + h*(8/9), U + h*(19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4)
    k6 = F(t1 + h, U + h*(9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5)
    k7 = F(t1 + h, U + h*(35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)

    U5 = U + h*((35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)
    U4 = U + h*((5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4 - (92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7)

    return U5


# CASH KARP: 6 ETAPAS
def RK_CashKarp(U, t1, t2, F):
    h = t2 - t1

    k1 = F(t1, U)
    k2 = F(t1 + h*(1/5), U + h*((1/5)*k1))
    k3 = F(t1 + h*(3/10), U + h*((3/40)*k1 + (9/40)*k2))
    k4 = F(t1 + h*(3/5), U + h*((3/10)*k1 - (9/10)*k2 + (6/5)*k3))
    k5 = F(t1 + h*(1), U + h*((-11/54)*k1 + (5/2)*k2 - (70/27)*k3 + (35/27)*k4))
    k6 = F(t1 + h*(7/8), U + h*((1631/55296)*k1 + (175/512)*k2 + (575/13824)*k3 + (44275/110592)*k4 + (253/4096)*k5))

    U5 = U + h*((37/378)*k1 + (0)*k2 + (250/621)*k3 + (125/594)*k4 + (0)*k5 + (512/1771)*k6)
    U4 = U + h*((2825/27648)*k1 + (0)*k2 + (18575/48384)*k3 + (13525/55296)*k4 + (277/14336)*k5 + (1/4)*k6)

    return U5


### 2) Write function to simulate the circular restricted three body problem
def CR_Three_Body_Problem(t, U, mu):
    x = U[0]
    y = U[1]
    vx = U[2]
    vy = U[3]

    r1 = sqrt((x + mu)**2 + y**2)
    r2 = sqrt((x - 1 + mu)**2 + y**2)

    ax = x + 2*vy - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    ay = y - 2*vx - (1 - mu)*y/r1**3 - mu*y/r2**3

    return array([vx, vy, ax, ay]) 


def RK_CRTBP(U0, tf, mu, method):
    N = 3000

    t = linspace(0, tf, N)
    U = zeros((N, len(U0)))     
    U[0] = U0

    def F(t2, U2):
        return CR_Three_Body_Problem(t2, U2, mu)

    for i in range(N - 1):
        t0 = t[i]
        t1 = t[i + 1]

        if method == "CASH KARP":
            U[i+1] = RK_CashKarp(U[i], t0, t1, F)

        elif method == "FEHLBERG":
            U[i+1] = RK_Fehlberg(U[i], t0, t1, F)

        elif method == "DORMAND PRINCE":
            U[i+1] = RK_DormandPrince(U[i], t0, t1, F)

    return U

### 3) Determination of the Lagrange points F(U) = 0
def Lagrange_System(V, mu):
    x = V [0]
    y = V [1]

    r1 = sqrt((x + mu)**2 + y**2)
    r2 = sqrt((x - 1 + mu)**2 + y**2)

    Fx = x - (1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3
    Fy = y - ((1 - mu)/r1**3 + mu/r2**3)*y

    return array([Fx, Fy])


def Lagrange_Points(mu):
    # Sol - Tierra
    Puntos_Lagrange_SolTierra = [
        (0.99, 0),              # L1 
        (1.01, 0),              # L2 
        (-1, 0),                # L3 
        (0.5, 0.86602540378),   # L4 
        (0.5, -0.86602540378)   # L5
    ]

    L = []
    for p in Puntos_Lagrange_SolTierra:
        solution = fsolve(Lagrange_System, p, args=(mu))
        L.append(solution)

    return L


mu = 3e-6
posicion_inicial = [1.02, 0, 0, 0.01]

# MÉTODO DE CASH KARP
U_CK = RK_CRTBP(posicion_inicial, tf=20, mu=mu, method="CASH KARP")

plt.figure()
plt.plot(U_CK[:,0], U_CK[:,1], label="Órbita")
L = array(Lagrange_Points(mu))
plt.scatter(L[:,0], L[:,1], c="red", marker="x", label="Puntos de Lagrange")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Runge Kutta de Cash - Karp")
plt.legend()
plt.show()

# MÉTODO DE FEHLBERG
U_F45 = RK_CRTBP(posicion_inicial, tf=20.0, mu=mu, method="FEHLBERG")

plt.figure()
plt.plot(U_F45[:,0], U_F45[:,1], label="Órbita")
L = array(Lagrange_Points(mu))
plt.scatter(L[:,0], L[:,1], c="red", marker="x", label="Puntos de Lagrange")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Runge Kutta de Fehlberg")
plt.legend()
plt.show()

# MÉTODO DE DORMAND PRINCE 
U_DP = RK_CRTBP(posicion_inicial, tf=20.0, mu=mu, method="DORMAND PRINCE")

plt.figure()
plt.plot(U_DP[:,0], U_DP[:,1], label="Órbita")
L = array(Lagrange_Points(mu))
plt.scatter(L[:,0], L[:,1], c="red", marker="x", label="Puntos de Lagrange")

plt.axis("equal")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Runge Kutta de Dormand - Prince")
plt.legend()
plt.show()
