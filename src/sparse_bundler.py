import numpy as np
from scipy import sparse
from global_map import LocalMap
from optimizer import Optimizer
from quarternion import Quarternion
from jacobian import derr_over_dcam, derr_over_dpw, derr_over_df


def inv_hpp_mat(mat, block_size=3):
    m = mat.shape[0]
    data, indices, indptr = [], [], []
    for i in range(0, m, block_size):
        block = mat[i: i + block_size, i: i + block_size].toarray()
        data.append(np.linalg.inv(block))
        indices.append(i // block_size)
        indptr.append(i // block_size)
    indptr.append(m // block_size)
    inv_mat = sparse.bsr_matrix(
                (np.asarray(data), np.asarray(indices), np.asarray(indptr)),
                shape=(m, m),
                blocksize=(block_size, block_size)
              )
    return inv_mat


def solve_block_equation(A, b):
    hcc, hcp, hpc, hpp = A
    bc, bp = b
    if hcc is None:
        hpp_inv = inv_hpp_mat(hpp.tolil(), block_size=3)
        dxp = -hpp_inv * bp
        return dxp
    elif hpp is None:
        hcc_inv = inv_hpp_mat(hcc.tolil(), block_size=7)
        dxc = -hcc_inv * bc
        return dxc
    else:
        hpp_inv = inv_hpp_mat(hpp.tolil(), block_size=3)
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

        self.cam_block_size = (2, 7)
        self.point_block_size = (2, 3)
        self.radius = 2e-3
        self.rpj_err = None
        self.loss = np.Inf
        self.max_try = 5
        self.max_iteration = 15

        # mode: {cam_opt_mode; point_opt_mode; full_ba}
        self.mode = "full_ba"

    def get_local_map_in_window(self, **kwargs):
        pw_index = kwargs.get("pw_index", None)
        if pw_index is not None:
            self.local_map = LocalMap(self.global_map, pw_index=pw_index)
            self.mode = "point_opt_mode"
        else:
            window = kwargs.get("window", self.global_map.window)
            self.local_map = LocalMap(self.global_map, window=window)

        self.fixed_pt_num = len(self.local_map.pw_index)
        self.fixed_frm_num = len(self.local_map.window)
        if self.fixed_frm_num == 0 and self.fixed_pt_num == 0:
            return False
        self.window_index = {}
        for i, frm_idx in enumerate(self.local_map.window):
            self.window_index[frm_idx] = i
        return True

    def jacobian_for_cam(self):
        landmark_idx = 0
        jc_data, indices_c = [], []
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
                # jfuv = derr_over_df(q, t, pt)
                # self.jc_data.append(np.column_stack((jpose, jfuv)))
                jc_data.append(jpose)
                indices_c.append(j)
                landmark_idx += 1

        M = landmark_idx * 2
        Nc = self.fixed_frm_num * self.cam_block_size[1]
        jc = sparse.bsr_matrix(
            (np.asarray(jc_data), np.asarray(indices_c), np.asarray(range(landmark_idx + 1))),
            shape=(M, Nc),
            blocksize=self.cam_block_size
        )
        return jc

    def jacobian_for_pt(self):
        landmark_idx = 0
        jp_data, indices_p = [], []
        for i, p_idx in enumerate(self.local_map.pw_index):
            pt = self.global_map.pw[p_idx]
            for frm_idx in self.global_map.viewed_frames[p_idx]:
                if self.global_map.frames[frm_idx].status is False:
                    continue
                j = self.window_index.get(frm_idx, None)
                if j is None and self.fixed_frm_num > 0:
                    continue
                frm = self.global_map.frames[frm_idx]
                q = Quarternion.mat_to_quaternion(frm.cam.R)
                fu, fv, t = frm.cam.fx, frm.cam.fy, frm.cam.t

                jpt = derr_over_dpw(q, t, fu, fv, pt)
                jp_data.append(jpt)
                indices_p.append(i)
                landmark_idx += 1

        M = landmark_idx * 2
        Np = self.fixed_pt_num * self.point_block_size[1]
        jp = sparse.bsr_matrix(
            (np.asarray(jp_data), np.asarray(indices_p), np.asarray(range(landmark_idx + 1))),
            shape=(M, Np),
            blocksize=self.point_block_size
        )
        return jp

    def calc_jacobian_mat(self):
        self.jc = self.jp = None
        if self.fixed_pt_num > 0 and self.fixed_frm_num != 1:
            self.jp = self.jacobian_for_pt()
        if self.fixed_frm_num > 0:
            self.jc = self.jacobian_for_cam()

        if self.jc is None:
            self.j_sparse = self.jp.tolil()
        elif self.jp is None:
            self.j_sparse = self.jc.tolil()
        else:
            self.j_sparse = sparse.hstack((self.jc, self.jp)).tolil()
        return self.j_sparse

    def calc_reprojection_err(self, var=None):
        if var is not None:
            bak = self.local_map.get_variables()
            self.local_map.set_variables(var)
        rpj_err = []
        for pw_idx in self.local_map.pw_index:
            for frm_idx in self.global_map.viewed_frames[pw_idx]:
                if self.global_map.frames[frm_idx].status is False:
                    continue
                if frm_idx not in self.window_index and self.fixed_frm_num > 0:
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
        self.hpp = self.hcc = self.hcp = self.hpc = None
        if self.jp is not None:
            self.hpp = self.jp.transpose() * self.jp
        if self.jc is not None:
            self.hcc = self.jc.transpose() * self.jc
        if self.jc is not None and self.jp is not None:
            self.hcp = self.jc.transpose() * self.jp
            self.hpc = self.hcp.transpose()

    def add_diagonal(self, hxx, radius):
        if hxx is not None:
            hxx = hxx + radius * sparse.eye(hxx.shape[0])
        return hxx

    def solve_lm(self, **kwargs):
        if self.get_local_map_in_window(**kwargs) is False:
            return
        self.radius = 2e-3   # reset
        iteration = 0
        while True:
            iteration += 1
            self.calc_jacobian_mat()
            self.calc_block_hessian_mat()
            self.rpj_err, self.loss = self.calc_reprojection_err()
            print("loss = %.5f, radius = %f" % (self.loss, self.radius))

            bp = bc = None
            if self.jc is not None:
                bc = self.jc.transpose() * self.rpj_err
            if self.jp is not None:
                bp = self.jp.transpose() * self.rpj_err
            var_bak = self.local_map.get_variables()

            try_iter = 0
            terminate_flag = True
            while try_iter < self.max_try:
                try_iter += 1
                hcc = self.add_diagonal(self.hcc, self.radius)
                hpp = self.add_diagonal(self.hpp, self.radius)

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

    def solve(self, max_iter=15, filter_err=True, **kwargs):
        if filter_err:
            self.max_iteration = max_iter * 0.7
        else:
            self.max_iteration = max_iter

        self.solve_lm(**kwargs)

        if filter_err:
            self.local_map.filter_error_points()
            self.max_iteration = max_iter * 0.3
            self.solve_lm()
