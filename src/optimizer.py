import geometry as geo
import numpy as np


class GaussNewton(object):
    def __init__(self, coef, data, target):
        self.lr = 0.01
        self.coef = coef
        self.data = data
        self.target = target
        self.residual = 0

    def forward(self, *args):
        pass

    def calc_jacobian_mat(self, *args):
        pass


class EpnpSolver(GaussNewton):
    def __init__(self, coef, data, target):
        super(EpnpSolver, self).__init__(coef, data, target)
        self.forward()

    def forward(self):
        N = len(self.data)
        self.eij, self.Mij = np.zeros((geo.sum_to_n(N - 1), 1)), []
        k, residual = 0, 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                Aij = self.data[i] - self.data[j]
                M = np.matmul(Aij.T, Aij)
                err = geo.quadratic_form(self.coef, M, self.coef) - self.target[k]
                self.Mij.append(M)
                self.eij[k] = err
                residual += err ** 2
                k += 1
        self.residual = residual
        return residual

    def calc_jacobian_mat(self):
        N = len(self.data)
        k, Jij = 0, np.zeros((geo.sum_to_n(N - 1), 4))
        for i in range(N):
            for j in range(i + 1, N):
                J = 2 * self.eij[k] * np.matmul((self.Mij[k] + self.Mij[k].T), self.coef)
                Jij[k, :] = J
                k += 1
        self.Jij = Jij
        return Jij

    def sovle(self):
        J = self.calc_jacobian_mat()
        H = np.matmul(J.T, J)
        b = np.squeeze(-np.matmul(J.T, self.eij ** 2))
        H_ = np.linalg.pinv(H)
        step = np.matmul(H_, b)
        self.coef += step
        self.forward()  # update residual with new coef
