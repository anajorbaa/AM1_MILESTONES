from numpy import zeros, array, linspace, sqrt
import matplotlib.pyplot as plt

T = 20
N = 2000

dt = T/N
#t = linspace(t0, tf, N+1)

# 1) EULER
U = zeros([N+1,4])
U[0,:] = array([1,0,0,1])

for n in range(N):
    x  = U[n, 0]
    y  = U[n, 1]
    vx = U[n, 2]
    vy = U[n, 3]

    r = sqrt(x**2+y**2)

    U[n+1, 0] = x + dt*vx
    U[n+1, 1] = y + dt*vy
    U[n+1, 2] = vx + dt*(- x / r**3)
    U[n+1, 3] = vy + dt*( - y / r**3)

# 2) CRANK-NICOLSON
U_CN = zeros([N+1,4])
U_CN[0,:] = array([1,0,0,1])

for n in range(N):
    x = U_CN[n, 0]
    y = U_CN[n, 1]
    vx = U_CN[n, 2]
    vy = U_CN[n, 3]
    
    r = sqrt(x**2+y**2)

    U_CN[n+1, 0] = x + dt * vx + 0.5 * dt**2 * (- x / r**3)
    U_CN[n+1, 1] = y + dt * vy + 0.5 * dt**2 * (- y / r**3)
    
    x_2 = U_CN[n+1, 0]
    y_2 = U_CN[n+1, 1]

    r_2 = sqrt(x_2**2+y_2**2)

    U_CN[n+1, 2] = vx + 0.5 * dt * ((- x / r**3) + (- x_2 / r_2**3))
    U_CN[n+1, 3] = vy + 0.5 * dt * ((- y / r**3) + (- y_2 / r_2**3))

# 3) RUNGE KUTA 4 
U_RK4 = zeros([N+1,4])
U_RK4[0,:] = array([1,0,0,1]) 

for n in range (N):
    x  = U_RK4[n, 0]
    y  = U_RK4[n, 1]
    vx = U_RK4[n, 2]
    vy = U_RK4[n, 3]

    r = sqrt(x**2+y**2)

    k1_x = vx
    k1_y = vy
    k1_vx = - x / r**3
    k1_vy= - y / r**3

    k2_x = vx + 0.5 * dt * k1_vx
    k2_y = vy + 0.5 * dt * k1_vy
    k2_vx = k1_vx + 0.5 * dt * k1_vx
    k2_vy = k1_vy + 0.5 * dt * k1_vy

    k3_x = vx + 0.5 * dt * k2_vx
    k3_y = vy + 0.5 * dt * k2_vy
    k3_vx = k1_vx + 0.5 * dt * k2_vx
    k3_vy = k1_vy + 0.5 * dt * k2_vy

    k4_x = vx + dt * k3_vx
    k4_y = vy + dt * k3_vy
    k4_vx = k1_vx + dt * k3_vx
    k4_vy = k1_vy + dt * k3_vy

    #RK4
    U_RK4[n+1, 0] = U_RK4[n, 0] + (dt/6)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
    U_RK4[n+1, 1] = U_RK4[n, 1] + (dt/6)*(k1_y + 2*k2_y + 2*k3_y + k4_y)
    U_RK4[n+1, 2] = U_RK4[n, 2] + (dt/6)*(k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
    U_RK4[n+1, 3] = U_RK4[n, 3] + (dt/6)*(k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)


plt.title("Órbitas de Kepler")
plt.plot(U[:, 0], U[:, 1], label="Euler")
plt.plot(U_CN[:, 0], U_CN[:, 1], label="Crank-Nickolson")
plt.plot(U_RK4[:, 0], U_RK4[:, 1], label="Runge-Kutta 4")
plt.legend(loc="upper right")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()    

"""Change time step and plot orbits. Discuss results
1) Euler: órbita en espiral 
2) Crank-Nicolson: órbita cerrada
3) RK4: órbita en espiral

Para:
a) N = 200
b) N = 1000
c) N = 2000

Al ir aumentando N las órbitas tienden a órbitas circulares.
"""