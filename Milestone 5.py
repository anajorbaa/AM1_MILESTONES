from numpy import zeros, concatenate, linspace
import matplotlib.pyplot as plt
from Modules.temporal_schemes import Cauchy, Crank_Nicolson, Euler, RK4, Inverse_Euler
from Modules.dynamic_functions import F
G = 1

def N_Body_Problem(masas):
    
    N_cuerpos = len (masas)

    def F_N_Body (U,t):
            
            x = U[:2*N_cuerpos].reshape((N_cuerpos, 2))     
            v = U[2*N_cuerpos:].reshape((N_cuerpos, 2))
            a = zeros((N_cuerpos, 2))

            for i in range(N_cuerpos):
                for j in range(N_cuerpos):
                    if i == j:
                        continue
                    
                    dx = x[j, 0] - x[i, 0]
                    dy = x[j, 1] - x[i, 1]

                    r2 = dx*dx + dy*dy      
                    if r2 == 0:
                        continue               
            
                    a[i, 0] = a[i, 0] + (G * masas[j] / (r2 * r2**0.5 )) * dx
                    a[i, 1] = a[i,1] + (G * masas[j] / (r2 * r2**0.5 )) * dy

            return concatenate((v.reshape(-1), a.reshape(-1)))
    return F_N_Body



# SIMULACIÓN 1: DOS CUERPOS
masas1 = [3, 0.001]

r1_0 = [0, 0]
r2_0 = [1, 0]

v1_0 = [0, 0]
v2_0 = [0, 1]

U0 = concatenate((r1_0, r2_0, v1_0, v2_0))

t0 = 0
tf = 50
N = 5000
t = linspace(t0, tf, N)

F_caso1 = N_Body_Problem(masas1)
U_caso1 = Cauchy(F_caso1, U0, t, RK4)

N_cuerpos_1 = len(masas1)
X_caso1 = U_caso1[:, :2*N_cuerpos_1].reshape((len(t), N_cuerpos_1, 2))

plt.figure()
for i in range(N_cuerpos_1):
    plt.plot(X_caso1[:, i, 0], X_caso1[:, i, 1], label=f'Cuerpo {i+1}')
    plt.scatter(X_caso1[0, i, 0], X_caso1[0, i, 1])  
plt.axis('equal')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Problema de N cuerpos: 2 cuerpos')
plt.show()

# SIMULACIÓN 2: TRES CUERPOS
masas2 = [1, 1, 1]

r1_0_2 = [-0.5,  0]
r2_0_2 = [ 0.5,  0]
r3_0_2 = [ 0,  0.8]

v1_0_2 = [ 0, -0.3]
v2_0_2 = [ 0,  0.3]
v3_0_2 = [ 0,  0]

U0_2 = concatenate((r1_0_2, r2_0_2, r3_0_2, v1_0_2, v2_0_2, v3_0_2))

F_caso2 = N_Body_Problem(masas2)
U_caso2 = Cauchy(F_caso2, U0_2, t, RK4)

N_cuerpos_2 = len(masas2)
X_caso2 = U_caso2[:, :2*N_cuerpos_2].reshape((len(t), N_cuerpos_2, 2))

plt.figure()
for i in range(N_cuerpos_2):
    plt.plot(X_caso2[:, i, 0], X_caso2[:, i, 1], label=f'Cuerpo {i+1}')
    plt.scatter(X_caso2[0, i, 0], X_caso2[0, i, 1]) 
plt.axis('equal')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Problema de N cuerpos: 3 cuerpos')
plt.show()

