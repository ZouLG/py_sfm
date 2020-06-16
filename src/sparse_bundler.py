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
        self.j_sparse = dict()
        self.h_sparse = dict()

    def __calc_sparse_jacobian__(self, frm):
        frm_num = len(self.graph.frames)
        for i in frm.kps_idx:
            if i is not np.Inf and self.graph.pw[i] is not None:
                q = Quarternion.mat_to_quaternion(frm.cam.R)
                fu = frm.cam.K[0, 0]
                fv = frm.cam.K[1, 1]
                jcam = derr_over_dcam(q, frm.cam.t, fu, fv, self.graph.pw[i])
                self.j_sparse[(self.attribute_idx, self.landmark_idx)] = jcam

                jpoint = derr_over_dpw(q, frm.cam.t, fu, fv, self.graph.pw[i])
                self.j_sparse[(i + frm_num, self.landmark_idx)] = jpoint
                self.landmark_idx += 1
        self.attribute_idx += 1

    def calc_jacobian_mat(self):
        self.landmark_idx = 0
        self.attribute_idx = 0
        for k, frm in enumerate(self.graph.frames):
            if frm.status is True:
                self.__calc_sparse_jacobian__(frm)

    def __calc_hcc__(self):
        frm_num = len(self.graph.frames)
        hcc = [np.zeros(7, 7)] * frm_num
        for j in self.j_sparse:
            if j[0] >= frm_num:
                continue
            j_mat = self.j_sparse[j]
            hcc[j[0]] += np.matmul(j_mat.T, j_mat)
        return hcc

    def __calc_hpp__(self):
        frm_num = len(self.graph.frames)
        point_num = len(self.graph.pw)
        hpp = [np.zeros(3, 3)] * point_num
        for j in self.j_sparse:
            if j[0] < frm_num:
                continue
            j_mat = self.j_sparse[j]
            hpp[j[0] - frm_num] += np.matmul(j_mat.T, j_mat)
        return hpp

    def __calc_hcp__(self):
        pass

    def calc_block_hessian_mat(self):
        self.h_sparse['hpp'] = self.__calc_hpp__()
        self.h_sparse['hcc'] = self.__calc_hcc__()
        self.h_sparse['hcp'] = self.__calc_hcp__()
