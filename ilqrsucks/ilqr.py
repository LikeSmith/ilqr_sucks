"""
ilqr.py

Author: Kyle Crandall
Date: July 2020

iterativ Linear Quadratic Regulator implimentation
"""

import numpy as np

EPS = 1e-4

class iLQR(object):
    def __init__(
        self,
        f,
        f_x,
        f_u,
        Q,
        R,
        Q_f=None,
        N=None):
        self.f = f
        self.f_x = f_x
        self.f_u = f_u
        self.Q = Q
        self.R = R
        self.Q_f = Q_f
        self.N = N

        if self.Q_f is None:
            self.Q_f = self.Q
    
    def run(self, x_0, s, dt=1.0, verbose=True, epochs=100):
        t = np.arange(s)*dt
        n = self.Q.shape[0]
        m = self.R.shape[0]
        K = np.zeros((s, m, n))
        x = np.zeros((s+1, n))
        u = np.zeros((s, m))
        L = []
        l = np.zeros(s+1)
        running = True

        x[0, :] = x_0

        while running:
            l = np.zeros(s)
            # Forward Prop
            for i in range(s):
                u[i, :] = -np.matmul(K[i, :, :], x[i, :])
                x[i+1, :] = self.f(x[i, :], u[i, :])
                l[i] += np.matmul(x[i, :], np.matmul(self.Q, x[i, :]))
                l[i] += np.matmul(u[i, :], np.matmul(self.R, u[i, :]))
                if self.N is not None:
                    l[i] += 2*np.matmul(x[i, :], np.matmul(self.N, u[i, :]))

            l[-1] += np.matmul(x[-1, :], np.matmul(self.Q_f, x[-1, :]))
            L.append(np.sum(l))

            # Backwards Pass
            P = self.Q_f
            for i in range(s):
                A = self.f_x(x[-i-1, :], u[-i-1, :])
                B = self.f_u(x[-i-1, :], u[-i-1, :])

                M1 = np.linalg.inv(self.R + np.matmul(B.T, np.matmul(P, B)))
                M2 = np.matmul(B.T, np.matmul(P, A))
                if self.N is not None:
                    M2 += self.N.T
                
                K[-i-1, :, :] = np.matmul(M1, M2)
                P = np.matmul(A.T, np.matmul(P, A))
                P -= np.matmul(M2.T, np.matmul(M1, M2))
                P += self.Q
            
            if verbose:
                print("Iteration %d complete, loss=%f"%(len(L), L[-1]))
            
            if len(L) > 2 and np.abs(L[-1] - L[-2]) < EPS or len(L) == epochs:
                running = False
                if verbose:
                    print("iLQR complete")
        return K, L
