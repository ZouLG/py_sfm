from jacobian import derr_over_dcam, derr_over_dpw
from optimizer import Optimizer
from map import Map
import numpy as np
from quarternion import Quarternion
from scipy import sparse
from scipy.sparse import linalg


def inv_hpp_mat(mat):
    m = mat.shape[0]
    data, indices, indptr = [], [], []
    for i in range(0, m, 3):
        block = mat[i: i + 3, i: i + 3].toarray()
        data.append(np.linalg.inv(block))
        indices.append(i // 3)
        indptr.append(i // 3)
    indptr.append(m // 3)
    return sparse.bsr_matrix((np.asarray(data), np.asarray(indices), np.asarray(indptr)),
                             shape=(m, m), blocksize=(3, 3))


def solve_block_equation(A, b):
    hcc, hcp, hpc, hpp = A
    bc, bp = b
    # hpp_inv = linalg.inv(hpp.tocsc())
    hpp_inv = inv_hpp_mat(hpp.tolil())
    schur = hcc - hcp * hpp_inv * hpc
    b = hcp * hpp_inv * bp - bc
    dxc = np.matmul(np.linalg.pinv(schur.toarray()), b)
    dxp = -hpp_inv * (bp + hpc * dxc)
    return np.concatenate([dxc, dxp])


class SparseBa(Optimizer):
    def __init__(self, graph):
        assert isinstance(graph, Map)
        self.graph = graph
        self.cam_block_size = (2, 7)
        self.point_block_size = (2, 3)
        self.radius = 2e-4
        self.loss = np.Inf
        self.iter_num = 5

    def __calc_sparse_jacobian__(self, frm, frm_idx):
        for i in frm.kps_idx:
            if i is not np.Inf and self.graph.pw[i] is not None:
                q = Quarternion.mat_to_quaternion(frm.cam.R)
                fu = frm.cam.K[0, 0]
                fv = frm.cam.K[1, 1]

                jcam = derr_over_dcam(q, frm.cam.t, fu, fv, self.graph.pw[i])
                self.jc_data.append(jcam)
                self.indices_c.append(frm_idx)

                jpoint = derr_over_dpw(q, frm.cam.t, fu, fv, self.graph.pw[i])
                self.jp_data.append(jpoint)
                self.indices_p.append(i)

                self.indptr.append(self.landmark_idx)
                self.landmark_idx += 1

    def calc_jacobian_mat(self):
        self.landmark_idx = 0
        self.indptr = []
        self.jc_data, self.indices_c = [], []
        self.jp_data, self.indices_p = [], []
        frm_idx = 0
        for frm in self.graph.frames:
            if frm.status is True:
                self.__calc_sparse_jacobian__(frm, frm_idx)
                frm_idx += 1
        self.indptr.append(self.landmark_idx)

        M = self.landmark_idx * 2
        Nc = self.graph.fixed_frm_num * self.cam_block_size[1]
        Np = self.graph.fixed_pt_num * self.point_block_size[1]     # number of points with fixed coordinates
        self.jc = sparse.bsr_matrix((np.asarray(self.jc_data), np.asarray(self.indices_c),
                        np.asarray(self.indptr)), shape=(M, Nc), blocksize=self.cam_block_size)
        self.jp = sparse.bsr_matrix((np.asarray(self.jp_data), np.asarray(self.indices_p),
                        np.asarray(self.indptr)), shape=(M, Np), blocksize=self.point_block_size)
        self.j_sparse = sparse.hstack((self.jc, self.jp))

    def __calc_reprojection_err__(self, frm, rpj_err):
        for k, i in enumerate(frm.kps_idx):
            if i is not np.Inf and self.graph.pw[i] is not None:
                pi_, _ = frm.cam.project_world2image([self.graph.pw[i]])
                rpj_err.append(pi_ - frm.pi[k, :])

    def calc_reprojection_err(self, var=None):
        if var is not None:
            bak = self.graph.get_variables()
            self.graph.set_variables(var)
        rpj_err = []
        for frm in self.graph.frames:
            if frm.status is True:
                self.__calc_reprojection_err__(frm, rpj_err)
        rpj_err = np.squeeze(np.concatenate(rpj_err, axis=1))
        loss = np.linalg.norm(rpj_err) / len(rpj_err)
        if var is not None:
            self.graph.set_variables(bak)
        return rpj_err, loss

    def calc_block_hessian_mat(self):
        self.hpp = self.jp.transpose() * self.jp
        self.hcc = self.jc.transpose() * self.jc
        self.hcp = self.jc.transpose() * self.jp
        self.hpc = self.hcp.transpose()

    def solve(self):
        self.calc_jacobian_mat()
        self.calc_block_hessian_mat()
        self.rpj_err, self.loss = self.calc_reprojection_err()
        print("loss = %.5f" % self.loss)

        bc = self.jc.transpose() * self.rpj_err
        bp = self.jp.transpose() * self.rpj_err
        dx = solve_block_equation([self.hcc, self.hcp, self.hpc, self.hpp], [bc, bp])
        var = self.graph.get_variables()
        var += dx
        self.graph.set_variables(var)

    def solve_lm(self):
        self.radius = 2e-4   # reset
        iteration = 0
        while True:
            iteration += 1
            self.calc_jacobian_mat()
            self.calc_block_hessian_mat()
            self.rpj_err, self.loss = self.calc_reprojection_err()
            print("loss = %.5f" % self.loss)

            bc = self.jc.transpose() * self.rpj_err
            bp = self.jp.transpose() * self.rpj_err
            var_bak = self.graph.get_variables()

            try_iter = 0
            terminate_flag = True
            while try_iter < self.iter_num:
                try_iter += 1
                hcc = self.hcc + self.radius * sparse.eye(self.hcc.shape[0])
                hpp = self.hpp + self.radius * sparse.eye(self.hpp.shape[0])

                dx = solve_block_equation([hcc, self.hcp, self.hpc, hpp], [bc, bp])
                var = var_bak + dx
                rpj_err, loss = self.calc_reprojection_err(var)
                if self.loss - loss > 1e-3:     # converge condition
                    self.graph.set_variables(var)
                    terminate_flag = False
                    self.radius /= 10
                    break
                else:
                    self.radius *= 10

            if terminate_flag or iteration >= self.iter_num:
                break
