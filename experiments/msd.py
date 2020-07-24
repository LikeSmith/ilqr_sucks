"""
msd.py

Author: Kyle Crandall
Date: July 2020

run iLQR on a mass spring damper system
"""

import numpy as np
import matplotlib.pyplot as plt
from ilqrsucks.ilqr import iLQR

M = 10.0
K = 2.0
D = 1.0
DT = 0.001
T_F = 10.0

X0 = np.array([10.0, 0.0])

Q = np.diag([1.0, 0.1])
R = np.diag([0.01])

def f(x, u):
    p = x[0]
    v = x[1]
    f = u[0]

    a_new = (f - p*K - v*D)/M
    v_new = v + a_new*DT
    p_new = p + v_new*DT

    return np.array([p_new, v_new])

def f_x(x, u):
    A = np.zeros((2, 2))

    A[0, 0] = 1.0 - DT**2*K/M
    A[0, 1] = DT - DT**2*D/M
    A[1, 0] = -DT*K/M
    A[1, 1] = 1.0 - DT*D/M

    return A

def f_u(x, u):
    B = np.zeros((2, 1))

    B[0, 0] = DT**2/M
    B[1, 0] = DT/M

    return B

if __name__ == '__main__':
    ctrl = iLQR(f, f_x, f_u, Q, R)
    n_steps = int(T_F/DT)

    k, L = ctrl.run(X0, n_steps, dt=DT)

    t = np.arange(n_steps+1)*DT
    x = np.zeros((n_steps+1, 2))
    u = np.zeros((n_steps, 1))
    l = np.zeros(n_steps+1)

    x[0, :] = X0

    for i in range(n_steps):
        u[i, :] = -np.matmul(k[i, :, :], x[i, :])
        x[i+1, :] = f(x[i, :], u[i, :])
        l[i] += np.matmul(x[i, :], np.matmul(Q, x[i, :]))
        l[i] += np.matmul(u[i, :], np.matmul(R, u[i, :]))
    l[-1] = np.matmul(x[-1, :], np.matmul(Q, x[-1, :]))

    plt.figure(1)
    plt.plot(L)
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Loss")
    
    plt.figure(2)
    plt.plot(t, x[:, 0])
    plt.xlabel("time (s)")
    plt.legend(["theta (rad)", "pos (m)"])

    plt.figure(3)
    plt.plot(t, x[:, 1])
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

    plt.figure(6)
    plt.subplot(121)
    plt.plot(t[:-1], k[:, 0, 0])
    plt.subplot(122)
    plt.plot(t[:-1], k[:, 0, 1])

    plt.show()
