from camera import *
from frame import Frame
from utils import *
import epnp


class Map():
    def __init__(self, *args, **kwargs):
        self.pw = []

        self.frames = []    # list of frames
        self.best_match = []
        self.detector = cv2.xfeatures2d_SIFT.create()
        self.scale = 10
        self.pj_err_th = 40
        self.total_err = 0

    def get_corresponding_matches(self, ref, mat):
        idx0, idx1, idx2 = [], [], []
        for i in range(len(mat.kps_idx)):
            k = mat.kps_idx[i]
            j = binary_search(ref.kps_idx, k)
            if j >= 0:
                idx0.append(k)
                idx1.append(j)
                idx2.append(i)
        pi0 = get_point_by_idx(ref.pi, idx1)
        pi1 = get_point_by_idx(mat.pi, idx2)
        return pi0, pi1, idx0

    def get_epnp_points(self, frm):
        idx = []
        pw = []
        for i in range(len(frm.kps_idx)):
            if self.pw[frm.kps_idx[i]] is not None:
                idx.append(i)
                pw.append(self.pw[frm.kps_idx[i]])
        return pw, frm.pi[idx, :]

    def bundle_adjust_with_2frms(self, ref, mat, t_scale, iter=20):
        if ref.status is False:
            return False
        pi0, pi1, idx = self.get_corresponding_matches(ref, mat)
        pc0 = ref.cam.project_image2camera(pi0)
        pc1 = mat.cam.project_image2camera(pi1)
        E, _ = get_null_space_ransac(list2mat(pc0), list2mat(pc1), max_iter=10)
        R_list, t_list = decompose_essential_mat(E)
        R, t = check_validation_rt(R_list, t_list, pc0, pc1)
        mat.cam.R = np.matmul(ref.cam.R, R)
        mat.cam.t = np.matmul(ref.cam.R, t)
        mat.cam.t = mat.cam.t / np.linalg.norm(mat.cam.t) * t_scale
        _, pw, _ = camera_triangulation(ref.cam, mat.cam, pi0, pi1)
        err_min = ref.cam.calc_projection_error(pw, pi0) + mat.cam.calc_projection_error(pw, pi1)
        pw_best = pw

        ratio = 0.8
        for i in range(iter):
            R, t = epnp.estimate_pose_epnp(mat.cam.K, pw, pi1)
            cam_tmp = PinHoleCamera(R, t, mat.cam.f)
            _, pw0, pw1 = camera_triangulation(ref.cam, cam_tmp, pi0, pi1)
            pw = [p * ratio + q * (1 - ratio) for p, q in zip(pw0, pw1)]
            err = ref.cam.calc_projection_error(pw, pi0) + mat.cam.calc_projection_error(pw, pi1)
            if err < err_min:
                err_min = err
                mat.cam = cam_tmp
                pw_best = pw

        if err_min > self.pj_err_th:
            print("Warning: projecting error is too big! frame skipped")
            mat.status = False
            return False
        else:
            for k, i in enumerate(idx):
                self.pw[i] = pw_best[k]
            mat.status = True
            return True

    def add_a_frame(self, frm, *args):
        # detect & match kps of the frm with the map
        if len(args) > 0 and isinstance(args[0], np.ndarray):
            frm.pi, frm.des = frm.detect_kps(args[0], self.detector)
            frm.kps_idx = [None] * len(frm.des)

        # if is the first frame
        self.best_match.append((None, 0))
        if len(self.frames) == 0:
            self.frames.append(frm)
            self.pw = [None] * len(frm.des)
            frm.kps_idx = list(range(len(frm.des)))
            frm.cam.R = np.eye(3)
            frm.cam.t = np.zeros((3, ))
            frm.status = True
            return

        # match new frame with all old frames to register each key point
        for i in range(len(self.frames)):
            ref = self.frames[i]
            idx0, idx1 = Frame.flann_match_kps(np.array(ref.des), np.array(frm.des))
            pi0 = get_point_by_idx(ref.pi, idx0)
            pi1 = get_point_by_idx(frm.pi, idx1)
            _, _, inliers = Frame.ransac_estimate_pose(pi0, pi1, ref.cam, frm.cam)
            if len(inliers) > self.best_match[i][1]:
                self.best_match[-1] = (i, len(inliers))
            for i in inliers:
                frm.kps_idx[idx1[i]] = ref.kps_idx[idx0[i]]

        # add mis-matched points to the map
        k = len(self.pw)
        for i, idx in enumerate(frm.kps_idx):
            if idx is None:
                frm.kps_idx[i] = k
                k += 1
                self.pw.append(None)

        # sort the kps by its index in the map
        frm.sort_kps_by_idx()

        # estimate pose of the frame
        pw, pi = self.get_epnp_points(frm)
        if len(pw) > 8:
            # frm.cam.R, frm.cam.t, inliers = epnp.ransac_estimate_pose(frm.cam.K, pw, pi)
            frm.cam.R, frm.cam.t = epnp.estimate_pose_epnp(frm.cam.K, pw, pi)
            err = frm.cam.calc_projection_error(pw, pi)
            if err > self.pj_err_th:      # set loose threshold
                print("Warning: projecting error is too big! frame skipped")
                frm.status = False
            else:
                frm.status = True
                frm.pj_err = err
        else:
            if self.best_match[-1][1] > 12:
                ref = self.frames[self.best_match[-1][0]]
                self.bundle_adjust_with_2frms(ref, frm, self.scale, iter=15)

        self.frames.append(frm)
        return

    def triangulate_2frms(self, frm1, frm2):
        if frm1.cam is None or frm2.cam is None:
            return
        pi1, pi2, idx = self.get_corresponding_matches(frm1, frm2)
        pw, pw1, pw2 = camera_triangulation(frm1.cam, frm2.cam, pi1, pi2)
        for i, j in enumerate(idx):
            if self.pw[j] is None:
                self.pw[j] = pw[i]
            else:
                self.pw[j] = (self.pw[j] + pw[i]) / 2

    def update_points(self):
        frm_num = len(self.frames)
        for i in range(0, frm_num - 1):
            ref = self.frames[i]
            for j in range(i + 1, frm_num):
                mat = self.frames[j]
                if ref.status is False or mat.status is False:
                    continue
                self.triangulate_2frms(ref, mat)

    def update_cam_pose(self):
        for frm in self.frames:
            pw, pi = self.get_epnp_points(frm)
            if len(pw) < 5:
                # frm.status = False
                continue
            frm.cam.R, frm.cam.t = epnp.estimate_pose_epnp(frm.cam.K, pw, pi)
            err = frm.cam.calc_projection_error(pw, pi)
            if err < self.pj_err_th:
                frm.status = True
                frm.pj_err = err
            else:
                print("Warning: projecting error is too big! frame skipped")
                # frm.status = False

    def reset_scale(self):
        """
            set the scale of the first best matching frame pair to self.scale and adjust the whole map
        """
        pass

    def calc_projecting_err(self):
        self.total_err = 0
        frm_num = 0
        for frm in self.frames:
            if frm.status is True:
                self.total_err += frm.pj_err
                frm_num += 1
        self.total_err /= (frm_num or 1)

    def plot_map(self, ax):
        for p in self.pw:
            if p is not None:
                p.plot3d(ax, marker='.', color='blue', s=10)

        for frm in self.frames:
            if frm.status is True:
                frm.cam.show(ax)
