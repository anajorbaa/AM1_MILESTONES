from numpy import zeros, linspace, array, log10
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Modules.temporal_schemes import Cauchy, Crank_Nicolson, Euler, RK4, Inverse_Euler
from Modules.dynamic_functions import F


U0 = ([1, 0, 0, 1])
t0 = 0
tf = 20
N = 10000
t = linspace (t0, tf, N + 1) 

# ERROR CAUCHY
def Cauchy_Error(F, U0, t, esquema, q):  

    N = len(t) - 1
    Error = zeros((N+1,len(U0)))

    t1 = t[:]
    t2 = linspace(t[0], t[N], 2*(N)+1 )

    U1 = Cauchy(F,U0,t1,esquema)  
    U2 = Cauchy(F,U0,t2,esquema)  

    for n in range(N+1):
        Error [n] = (U2 [2*n] - U1 [n]) / (1 - 1/2**q)

    return U1, Error

# GRÁFICAS
U1_E, Error_E = Cauchy_Error(F, U0, t, Euler, 1)
U1_IE, Error_IE = Cauchy_Error(F, U0, t, Inverse_Euler, 1)
U1_CN, Error_CN = Cauchy_Error(F, U0, t, Crank_Nicolson, 2)
U1_RK4, Error_RK4 = Cauchy_Error(F, U0, t, RK4, 4)

# ERROR 
plt.figure()
plt.plot(t, Error_E[:,1], label = "Euler")
plt.plot(t, Error_IE[:,1], label = "Euler inverso")
plt.plot(t, Error_CN[:,1], label = "Crank Nicolson")
plt.plot(t, Error_RK4[:,1], label = "RK4")

plt.xlabel('t')
plt.ylabel('Error')
plt.title('Error')
plt.grid(True)
plt.legend()

# ERROR ABSOLUTO
plt.figure()
plt.plot(t, abs(Error_E[:,1]),  label="Euler")
plt.plot(t, abs(Error_IE[:,1]), label="Euler inverso")
plt.plot(t, abs(Error_CN[:,1]), label="Crank Nicolson")
plt.plot(t, abs(Error_RK4[:,1]),label="RK4")

plt.title('Error absoluto')
plt.yscale('log')           
plt.xlabel('t')
plt.ylabel('|Error|')
plt.grid(True)
plt.legend()

# CONVERGENCIA
def Convergencia (esquema, F, U0, t, q):
    #mallas = array([10, 20, 40, 80])
    mallas = array([10, 20, 40, 80, 160, 320])
    N_m = len(mallas)

    logN = zeros(N_m)
    logE = zeros(N_m)

    for i in range(N_m):
        t_n = linspace(t[0], t[-1], mallas[i])
        U1, E = Cauchy_Error(F, U0, t_n, esquema, q)

        logN[i] = log10(mallas[i])
        logE[i] = log10(norm(E, axis=1).max())

    return logN, logE

# GRÁFICAS
logN_E, logE_E  = Convergencia (Euler, F, U0, t, 1)
logN_IE, logE_IE = Convergencia (Inverse_Euler, F, U0, t, 1)
logN_CN, logE_CN = Convergencia (Crank_Nicolson, F, U0, t, 2)
logN_RK4, logE_RK4 = Convergencia (RK4, F, U0, t, 4)

def pendiente(logN, logE):
    x = logN
    y = logE
    x_med = x.mean()
    y_med = y.mean()

    num = ((x - x_med) * (y - y_med)).sum()
    den = ((x - x_med)**2).sum()

    return num / den

pendiente_E = pendiente(logN_E,  logE_E)
pendiente_IE = pendiente(logN_IE, logE_IE)
pendiente_CN = pendiente(logN_CN, logE_CN)
pendiente_RK4 = pendiente(logN_RK4,logE_RK4)

print("Pendiente Euler = ", pendiente_E)
print("Pendiente Euler inverso = ", pendiente_IE)
print("Pendiente Crank Nicolson = ", pendiente_CN)
print("Pendiente RK4 = ", pendiente_RK4)

# CONVERGENCIA EULER
plt.figure()
plt.plot(logN_E, logE_E)
plt.title('Convergencia Euler')
plt.xlabel('log N')
plt.ylabel('log E')
plt.grid(True)

# CONVERGENCIA EULER INVERSO
plt.figure()
plt.plot(logN_IE, logE_IE)
plt.title('Convergencia Euler Inverso')
plt.xlabel('log N')
plt.ylabel('log E')
plt.grid(True)

# CONVERGENCIA CRANK NICOLSON
plt.figure()
plt.plot(logN_CN, logE_CN)
plt.title('Convergencia Crank Nicolson')
plt.xlabel('log N')
plt.ylabel('log E')
plt.grid(True)

# CONVERGENCIA RK4
plt.figure()
plt.plot(logN_RK4, logE_RK4)
plt.title('Convergencia RK4')
plt.xlabel('log N')
plt.ylabel('log E')
plt.grid(True)

plt.show()

### DISCUSIÓN

# ERROR
# 1) Euler y Euler inverso: la amplitud del error crece con el tiempo, el error va creciendo.
# 2) Crank Nicolson: mantiene un error muy pequeño y prácticamente 0.
# 3) RK4: el error es prácticamente nulo en todo el intervalo.

# ERROR ABSOLUTO
# 1) Euler: tiene el error más grande.
# 2) Euler inverso: el error sigue siendo bastante grande.
# 3) Crank Nicolson: reduce el error hasta valores de 10^-7 / 10^-8.
# 4) RK4: método más preciso.


# CONVERGENCIA 
# Euler explícito
# La pendiente obtenida es aproximadamente -0.44

# Euler inverso
# Euler inverso muestra una pendiente pequeña y positiva de 0.27, que no refleja correctamente su orden teórico 1. 

# Crank Nicolson
# La pendiente es de -2.28, muy cercana al orden 2 teórico.

# RK4
# Se obtiene una pendiente de -4.76, por encim del orden 4 teórico.
