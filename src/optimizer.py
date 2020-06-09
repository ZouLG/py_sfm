import geometry as geo
import numpy as np
from quarternion import Quarternion
from jacobian import derr_over_dcam


class Optimizer(object):
    def __init__(self):
        pass

    def forward(self, *args):
        pass

    def calc_jacobian_mat(self, *args):
        pass

    def solve(self):
        pass


class EpnpSolver(Optimizer):
    def __init__(self, coef, data, target):
        super(EpnpSolver, self).__init__()
        self.lr = 0.01
        self.coef = coef
        self.data = data
        self.target = target
        self.residual = 0
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


class PnpSolver(Optimizer):
    def __init__(self, pose, pw, pi, K):
        self.pw = pw
        self.pi = pi
        self.K = K
        self.fu = K[0, 0]
        self.fv = K[1, 1]
        self.quat = pose[0]
        self.t = pose[1]

    def forward(self):
        self._pi, self.pc = [], []
        self.err = np.zeros((len(self.pw) * 2,))
        for k, p in enumerate(self.pw):
            self.pc.append(Quarternion.rotate_with_quaternion(self.quat, p.p) + self.t)
            uv = np.matmul(self.K, self.pc[k])
            uv /= uv[2]
            self._pi.append(uv[0: 2])   # reprojected coordinate
            self.err[k * 2: k * 2 + 2] = self._pi[k] - self.pi[k]
        self.residual = np.matmul(self.err, self.err) / len(self.pw)
        return self.residual

    def calc_jacobian_mat(self):
        jcam = np.zeros((len(self.pw) * 2, 7))
        for k in range(len(self.pw)):
            pw, pi = self.pw[k], self.pi[k, :]
            jcam[2 * k: 2 * k + 2, :] = derr_over_dcam(self.quat, self.t, self.fu, self.fv, pw)
        return jcam

    def solve(self):
        self.forward()
        J = self.calc_jacobian_mat()
        H = np.matmul(J.T, J)
        Hinv = np.linalg.pinv(H)
        dx = np.matmul(np.matmul(Hinv, J.T), self.err)
        self.quat = self.quat - dx[0:4]
        self.t = self.t - dx[4:]


class SparseBa(Optimizer):
    def __init__(self):
        pass

    def calc_jacobian_mat(self, *args):
        pass