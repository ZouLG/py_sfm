import geometry as geo
import numpy as np
from quarternion import Quarternion


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


class PnpLmSolver(Optimizer):
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

    @staticmethod
    def derr_over_dpc(q, t, fu, fv, pw):
        pc = Quarternion.rotate_with_quaternion(q, pw.p) + t
        return np.array([[fu, 0, -fu * pc[0] / pc[2]],
                         [0, fv, -fv * pc[1] / pc[2]]]) / pc[2]

    @staticmethod
    def derr_over_dquat(q, t, fu, fv, pw):
        """
            2x4 jacobian matrix of derivative err to camera rotation
        """
        dedpc = PnpLmSolver.derr_over_dpc(q, t, fu, fv, pw)
        dpcdr = np.zeros((3, 9))
        dpcdr[0, 0:3] = pw.p
        dpcdr[1, 3:6] = pw.p
        dpcdr[2, 6:9] = pw.p
        q0, q1, q2, q3 = q
        drdq = 2 * np.array([[0, 0, -2 * q2, -2 * q3],
                             [-q3, q2, q1, -q0],
                             [q2, q3, q0, q1],
                             [q3, q2, q1, q0],
                             [0, -2 * q1, 0, -2 * q3],
                             [-q1, -q0, q3, q2],
                             [-q2, q3, -q0, q1],
                             [q1, q0, q3, q2],
                             [0, -2 * q1, -2 * q2, 0]])
        return np.matmul(np.matmul(dedpc, dpcdr), drdq)

    @staticmethod
    def derr_over_dt(q, t, fu, fv, pw):
        return PnpLmSolver.derr_over_dpc(q, t, fu, fv, pw)

    def calc_jacobian_mat_q(self):
        j = np.zeros((len(self.pw) * 2, 4))
        for k in range(len(self.pw)):
            pw, pi = self.pw[k], self.pi[k, :]
            j[2 * k: 2 * k + 2, :] = PnpLmSolver.derr_over_dquat(self.quat, self.t, self.fu, self.fv, pw)
        return j

    def calc_jacobian_mat_t(self):
        j = np.zeros((len(self.pw) * 2, 3))
        for k in range(len(self.pw)):
            pw, pi = self.pw[k], self.pi[k, :]
            j[2 * k: 2 * k + 2, :] = PnpLmSolver.derr_over_dt(self.quat, self.t, self.fu, self.fv, pw)
        return j

    def solve_q(self):
        self.forward()
        J = self.calc_jacobian_mat_q()
        H = np.matmul(J.T, J)
        Hinv = np.linalg.pinv(H)
        dq = np.matmul(np.matmul(Hinv, J.T), self.err)
        self.quat = self.quat - dq
        self.quat = self.quat / self.quat.norm()

    def solve_t(self):
        self.forward()
        J = self.calc_jacobian_mat_t()
        H = np.matmul(J.T, J)
        Hinv = np.linalg.pinv(H)
        dt = np.matmul(np.matmul(Hinv, J.T), self.err)
        self.t = self.t - dt
