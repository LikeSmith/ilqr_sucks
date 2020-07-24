"""
cartpole.py

Author: Kyle Crandall
Date: July 2020

run iLQR on cartpole system.
"""

import numpy as np
import matplotlib.pyplot as plt
from ilqrsucks.ilqr import iLQR

X0 = np.array([10.0*np.pi/180.0, 0.0, 0.0, 0.0])
M1 = 1.0
M2 = 0.5
G = 9.81
L = 1.0
DT = 0.001
T_F = 10.0
EPS = 1e-5

Q = np.diag([1.0, 0.1, 1.0, 0.1])
R = np.diag([0.01])

def f(x, u):
    theta = x[0]
    omega = x[2]
    p = x[1]
    v = x[3]
    f = u[0]

    M = np.array([[M1 + M2, -M2*L*np.cos(theta)], [-np.cos(theta), L]])
    x_ddot = np.matmul(np.linalg.inv(M), np.array([f - M2*L*omega**2, G*np.sin(theta)]))

    a_new = x_ddot[0]
    v_new = v + DT*a_new
    p_new = p + DT*v_new

    alpha_new = x_ddot[1]
    omega_new = omega + DT*alpha_new
    theta_new = theta + DT*omega_new
    
    theta_new = ((theta_new + np.pi) % (2*np.pi)) - np.pi

    if np.abs(p_new) > 10.0:
        p_new = 10.0*np.sign(p_new)
        v_new = 0.0

    return np.array([theta_new, omega_new, p_new, v_new])

def f_x(x, u):
    A = np.zeros((4, 4))
    x1 = f(x, u)

    for i in range(4):
        x_new = x.copy()
        x_new[i] += EPS

        A[i, :] = (f(x_new, u) - x1)/EPS
    
    return A

def f_u(x, u):
    B = np.zeros((4, 1))
    x1 = f(x, u)
    
    B[:, 0] = (f(x, u+EPS) - x1)/EPS

    return B

if __name__ == "__main__":
    ctrl = iLQR(f, f_x, f_u, Q, R)
    n_steps = int(T_F/DT)

    K, Losses = ctrl.run(X0, n_steps, dt=DT)

    t = np.arange(n_steps+1)*DT
    x = np.zeros((n_steps+1, 4))
    u = np.zeros((n_steps, 1))
    l = np.zeros(n_steps+1)

    x[0, :] = X0

    for i in range(n_steps):
        u[i, :] = -np.matmul(K[i, :, :], x[i, :])
        x[i+1, :] = f(x[i, :], u[i, :])
        l[i] += np.matmul(x[i, :], np.matmul(Q, x[i, :]))
        l[i] += np.matmul(u[i, :], np.matmul(R, u[i, :]))
    l[-1] = np.matmul(x[-1, :], np.matmul(Q, x[-1, :]))

    plt.figure(1)
    plt.plot(Losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Loss")
    
    plt.figure(2)
    plt.plot(t, x[:, [0, 2]])
    plt.xlabel("time (s)")
    plt.legend(["theta (rad)", "pos (m)"])

    plt.figure(3)
    plt.plot(t, x[:, [1, 3]])
    plt.xlabel("time (s)")
    plt.legend(["omega (rad/s)", "vel (m/s)"])

    plt.figure(4)
    plt.plot(t[:-1], u[:, 0])
    plt.xlabel("time (s)")
    plt.ylabel("f (N)")

    plt.figure(5)
    plt.plot(t, l)
    plt.xlabel("time(s)")
    plt.ylabel("loss")

    plt.show()
