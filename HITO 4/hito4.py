from numpy import linspace, array, sqrt, meshgrid
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from funciones import F, Cauchy, Crank_Nicolson, Euler, RK4, Inverse_Euler

from matplotlib.colors import ListedColormap
color_graficas = ListedColormap(["#d8e3e7", "#78eb81"])

U0 = array ([1, 0])
t0 = 0
tf = 50
N = 1000

t = linspace(t0, tf, N)

def Oscilador (U, t):
    x = U [0]
    v = U [1]
    F = array ([v, -x])
    return F

U_E = Cauchy (Oscilador, U0, t, Euler )
U_IE = Cauchy (Oscilador, U0, t, Inverse_Euler)
U_CN = Cauchy (Oscilador, U0, t, Crank_Nicolson )
U_RK4 = Cauchy (Oscilador, U0, t, RK4 )


fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# EULER
axs[0, 0].plot(U_E[:,0], U_E[:,1])
axs[0, 0].set_title("Euler")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("v")
axs[0, 0].axis("equal")
axs[0, 0].grid(True)

# EULER INVERSO
axs[0, 1].plot(U_IE[:,0], U_IE[:,1])
axs[0, 1].set_title("Euler Inverso")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("v")
axs[0, 1].axis("equal")
axs[0, 1].grid(True)

# CRANK NICOLSON
axs[1, 0].plot(U_CN[:,0], U_CN[:,1])
axs[1, 0].set_title("Crank Nicolson")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("v")
axs[1, 0].axis("equal")
axs[1, 0].grid(True)

# RK4
axs[1, 1].plot(U_RK4[:,0], U_RK4[:,1])
axs[1, 1].set_title("RK4")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("v")
axs[1, 1].axis("equal")
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()


# REGIONES DE ESTABILIDAD
def Est_Euler (omega):
    return 1 + omega

def Est_Inverse_Euler (omega):
    return 1 / (1 - omega)

def Est_Crank_Nicolson (omega):
    return (1 + 0.5*omega) / (1 - 0.5*omega)

def Est_RK4 (omega):
    return 1 + omega + 0.5*omega**2 + (1/6)*omega**3 + (1/24)*omega**4

def Est_LF (omega):
    a = 2 - omega**2
    b = a**2 - 4
    
    r1 = (a + sqrt(b)) / 2
    r2 = (a - sqrt(b)) / 2

    return r1, r2 

# FUNCION PLOT PARA REGIONES ESTABILIDAD
def plot_regiones_estabilidad():
    puntos = 300
    Re = linspace(-3, 3, puntos)
    Im = linspace(-3, 3, puntos)
    RE, IM = meshgrid(Re, Im)
    Z = RE + 1j*IM

    est_E = abs(Est_Euler(Z)) <= 1
    est_IE = abs(Est_Inverse_Euler(Z)) <= 1
    est_CN = abs(Est_Crank_Nicolson(Z)) <= 1
    est_RK4 = abs(Est_RK4(Z)) <= 1

    omega = IM  

    est_LF = (abs(omega) < 2) & (abs(RE) < 0.1)


    fig, axs = plt.subplots (2, 3, figsize = (12, 8))

    # EULER
    im0 = axs[0, 0].imshow(est_E.astype(int),
                       extent = [Re[0], Re[-1], Im[0], Im[-1]],
                       origin = 'lower',
                       aspect = 'equal',
                       cmap = color_graficas)
    axs[0, 0].set_title("Euler")
    axs[0, 0].set_xlabel("Re(z)")
    axs[0, 0].set_ylabel("Im(z)")

    # EULER INVERSO
    im1 = axs[0, 1].imshow(est_IE.astype(int),
                           extent = [Re[0], Re[-1], Im[0], Im[-1]],
                           origin = 'lower',
                           aspect = 'equal',
                           cmap = color_graficas)
    axs[0, 1].set_title("Euler inverso")
    axs[0, 1].set_xlabel("Re(z)")
    axs[0, 1].set_ylabel("Im(z)")

    # CRANK NICOLSON
    im2 = axs[0, 2].imshow(est_CN.astype(int),
                           extent = [Re[0], Re[-1], Im[0], Im[-1]],
                           origin = 'lower',
                           aspect = 'equal',
                           cmap = color_graficas)
    axs[0, 2].set_title("Crank Nicolson")
    axs[0, 2].set_xlabel("Re(z)")
    axs[0, 2].set_ylabel("Im(z)")

    # RK4
    im3 = axs[1, 0].imshow(est_RK4.astype(int),
                           extent = [Re[0], Re[-1], Im[0], Im[-1]],
                           origin = 'lower',
                           aspect = 'equal',
                           cmap = color_graficas)
    axs[1, 0].set_title("RK4")
    axs[1, 0].set_xlabel("Re(z)")
    axs[1, 0].set_ylabel("Im(z)")

    # LEAP FROG
    im4 = axs[1, 1].imshow(est_LF.astype(int),
                           extent = [Re[0], Re[-1], Im[0], Im[-1]],
                           origin = 'lower',
                           aspect = 'equal',
                           cmap = color_graficas)
    axs[1, 1].set_title("Leap Frog")
    axs[1, 1].set_xlabel("Re(z)")
    axs[1, 1].set_ylabel("Im(z)")
    
    axs[1, 2].axis("off")

    gris  = mpatches.Patch(color="#d8e3e7", label="Región Inestable")
    verde = mpatches.Patch(color="#78eb81", label="Región Estable")

    axs[1, 2].legend(handles=[gris, verde],
                    loc="center",
                    fontsize=11,
                    frameon=False)
    axs[1, 2].set_title("Leyenda")

    plt.tight_layout()
    plt.show()

plot_regiones_estabilidad()

# DISCUSIÓN
# 1) EULER: La región estable es pequeña y está situada en valores negativos de Re (z).
# 2) EULER INVERSO: Al contrario que el Euler, su región estable ocupa la mayor parte.
# 3) CRANK NICOLSON: Presenta estabilidad en el semiplano izquierdo.
# 4) RK4: Para este caso, la región de estabilidad es mayor que en los casos anteriores pero sigue sin cubrir todo el plano.
# 5) LEAP FROG: Este método presenta una región estable alrededor del eje imaginario. 