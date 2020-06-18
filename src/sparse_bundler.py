from jacobian import derr_over_dcam, derr_over_dpw
from optimizer import Optimizer
from map import Map
import numpy as np
from scipy import sparse
from quarternion import Quarternion


class SparseBa(Optimizer):
    def __init__(self, graph):
        assert isinstance(graph, Map)
        self.graph = graph
        self.cam_block_size = (2, 7)
        self.point_block_size = (2, 3)

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
        Nc = frm_idx * self.cam_block_size[1]
        Np = len(self.graph.pw) * self.point_block_size[1]
        self.jc = sparse.bsr_matrix((np.asarray(self.jc_data), np.asarray(self.indptr), np.asarray(self.indices_c)),
                                    shape=(M, Nc), blocksize=self.cam_block_size)
        self.jp = sparse.bsr_matrix((np.asarray(self.jp_data), np.asarray(self.indptr), np.asarray(self.indices_p)),
                                    shape=(M, Np), blocksize=self.point_block_size)
        self.j_sparse = sparse.hstack((self.jc, self.jp))

    def calc_block_hessian_mat(self):
        self.h_sparse = dict()
        self.h_sparse['hpp'] = self.jp.transpose() * self.jp
        self.h_sparse['hcc'] = self.jc.transpose() * self.jc
        self.h_sparse['hcp'] = self.jc.transpose() * self.jp
