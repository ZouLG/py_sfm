import numpy as np
from scipy import sparse
from global_map import LocalMap
from optimizer import Optimizer
from quarternion import Quarternion
from jacobian import derr_over_dcam, derr_over_dpw, derr_over_df


def inv_hpp_mat(mat):
    m = mat.shape[0]
    data, indices, indptr = [], [], []
    for i in range(0, m, 3):
        block = mat[i: i + 3, i: i + 3].toarray()
        data.append(np.linalg.inv(block))
        indices.append(i // 3)
        indptr.append(i // 3)
    indptr.append(m // 3)
    inv_mat = sparse.bsr_matrix(
                (np.asarray(data), np.asarray(indices), np.asarray(indptr)),
                shape=(m, m),
                blocksize=(3, 3)
              )
    return inv_mat


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
    def __init__(self, global_map):
        super(SparseBa, self).__init__()
        self.global_map = global_map
        self.local_map = None
        self.window_index = {}
        self.fixed_pt_num = 0
        self.fixed_frm_num = 0

        self.cam_block_size = (2, 9)
        self.point_block_size = (2, 3)
        self.radius = 2e-3
        self.rpj_err = None
        self.loss = np.Inf
        self.max_try = 5
        self.max_iteration = 15

    def get_local_map_in_window(self, window):
        self.local_map = LocalMap(self.global_map, window)
        self.window_index = {}
        for i, frm_idx in enumerate(self.local_map.window):
            self.window_index[frm_idx] = i
        self.fixed_pt_num = len(self.local_map.pw_index)
        self.fixed_frm_num = len(self.window_index)

    def calc_jacobian_mat(self):
        landmark_idx = 0
        self.indptr = []
        self.jc_data, self.indices_c = [], []
        self.jp_data, self.indices_p = [], []

        for i, p_idx in enumerate(self.local_map.pw_index):
            pt = self.global_map.pw[p_idx]
            for frm_idx in self.global_map.viewed_frames[p_idx]:
                j = self.window_index.get(frm_idx, None)
                if j is None:
                    continue
                frm = self.global_map.frames[frm_idx]
                q = Quarternion.mat_to_quaternion(frm.cam.R)
                fu, fv, t = frm.cam.fx, frm.cam.fy, frm.cam.t

                jpose = derr_over_dcam(q, t, fu, fv, pt)
                jfuv = derr_over_df(q, t, pt)
                self.jc_data.append(np.column_stack((jpose, jfuv)))
                self.indices_c.append(j)

                jpt = derr_over_dpw(q, t, fu, fv, pt)
                self.jp_data.append(jpt)
                self.indices_p.append(i)

                self.indptr.append(landmark_idx)
                landmark_idx += 1

        self.indptr.append(landmark_idx)

        M = landmark_idx * 2
        Nc = self.fixed_frm_num * self.cam_block_size[1]
        Np = self.fixed_pt_num * self.point_block_size[1]
        self.jc = sparse.bsr_matrix(
            (np.asarray(self.jc_data), np.asarray(self.indices_c), np.asarray(self.indptr)), 
            shape=(M, Nc),
            blocksize=self.cam_block_size
        )
        self.jp = sparse.bsr_matrix(
            (np.asarray(self.jp_data), np.asarray(self.indices_p), np.asarray(self.indptr)),
            shape=(M, Np),
            blocksize=self.point_block_size
        )
        self.j_sparse = sparse.hstack((self.jc, self.jp)).tolil()
        return self.j_sparse

    def calc_reprojection_err(self, var=None):
        if var is not None:
            bak = self.local_map.get_variables()
            self.local_map.set_variables(var)
        rpj_err = []
        for pw_idx in self.local_map.pw_index:
            for frm_idx in self.global_map.viewed_frames[pw_idx]:
                if frm_idx not in self.window_index:
                    continue
                frm = self.global_map.frames[frm_idx]
                pi_idx = frm.pw_pi[pw_idx]
                pi_, _ = frm.cam.project_world2image([self.global_map.pw[pw_idx]])
                rpj_err.append(pi_ - frm.pi[pi_idx, :])
        rpj_err = np.squeeze(np.concatenate(rpj_err, axis=1))
        loss = np.sqrt(np.matmul(rpj_err, rpj_err) / len(rpj_err) * 2)
        if var is not None:
            self.local_map.set_variables(bak)
        return rpj_err, loss

    def calc_linear_approximate_err(self, dx):
        err = self.rpj_err + self.j_sparse * dx
        loss = np.sqrt(np.matmul(err, err) / len(err) * 2)
        return err, loss

    def calc_gain(self, dx, loss):
        derr = self.loss - loss
        _, loss_linear = self.calc_linear_approximate_err(dx)
        derr_linear = self.loss - loss_linear
        ro = derr / derr_linear
        if ro < 0:
            return 5.0, False
        elif ro < 0.25:
            return 5.0, True
        elif ro > 0.75:
            return 0.2, True
        else:
            return 1.0, True

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
        var = self.local_map.get_variables()
        var += dx
        self.local_map.set_variables(var)

    def solve_lm(self, window=None):
        window = window or self.global_map.window
        self.get_local_map_in_window(window)
        self.radius = 2e-3   # reset
        iteration = 0
        while True:
            iteration += 1
            self.calc_jacobian_mat()
            self.calc_block_hessian_mat()
            self.rpj_err, self.loss = self.calc_reprojection_err()
            print("loss = %.5f, radius = %f" % (self.loss, self.radius))

            bc = self.jc.transpose() * self.rpj_err
            bp = self.jp.transpose() * self.rpj_err
            var_bak = self.local_map.get_variables()

            try_iter = 0
            terminate_flag = True
            while try_iter < self.max_try:
                try_iter += 1
                hcc = self.hcc + self.radius * sparse.eye(self.hcc.shape[0])
                hpp = self.hpp + self.radius * sparse.eye(self.hpp.shape[0])

                dx = solve_block_equation([hcc, self.hcp, self.hpc, hpp], [bc, bp])
                var = var_bak + dx
                rpj_err, loss = self.calc_reprojection_err(var)

                gain, status = self.calc_gain(dx, loss)
                if status:
                    self.local_map.set_variables(var)
                    self.radius *= gain
                    terminate_flag = (self.loss - loss) < 0.1  # converge flag
                    break
                else:
                    self.radius *= gain

            if terminate_flag or iteration >= self.max_iteration:
                break
