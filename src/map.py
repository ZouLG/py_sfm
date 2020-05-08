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

    def get_corresponding_matches(self, ref, mat):
        idx0 = []
        idx1 = []
        idx2 = []
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

    def get_exif(self):
        pass

    def bundle_adjust_with_2frms(self, ref, mat, iter=20):
        # get corresponding kps of the 2 frames
        i, j = 0, 0
        idx0, idx1 = [], []     # local index in a frame
        while i < len(ref.kps_idx) and j < len(mat.kps_idx):
            if ref.kps_idx[i] < mat.kps_idx[j]:
                i += 1
            elif ref.kps_idx[i] > mat.kps_idx[j]:
                j += 1
            else:
                idx0.append(i)
                idx1.append(j)
                i += 1
                j += 1
        pi0 = ref.pi[idx0, :]
        pi1 = mat.pi[idx1, :]

        ratio = 0.5
        for i in range(iter):
            _, pw1, pw2 = camera_triangulation(ref.cam, mat.cam, pi0, pi1)
            pw = [p * ratio + q * (1 - ratio) for p, q in zip(pw1, pw2)]
            R0, t0, _ = epnp.ransac_estimate_pose(ref.cam.K, pw, pi0, 10, 10)
            R1, t1, _ = epnp.ransac_estimate_pose(mat.cam.K, pw, pi1, 10, 10)
            ref.cam.R = R0
            ref.cam.t = t0
            mat.cam.R = R1
            mat.cam.t = t1
            for j in range(len(idx0)):
                q = self.pw[ref.kps_idx[idx0[j]]]
                if q is not None:
                    self.pw[ref.kps_idx[j]] = q * ratio + pw[j] * (1 - ratio)
                else:
                    self.pw[ref.kps_idx[j]] = pw[j]

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
        if len(pw) < 8:
            if self.best_match[-1][1] > 12:
                ref = self.frames[self.best_match[-1][0]]
                pw_, idx_ = frm.estimate_pose_and_points(ref)
                if len(pw_) == 0:
                    return

                for i in range(len(idx_)):
                    k = frm.kps_idx[idx_[i]]
                    self.pw[k] = pw_[i]
                # self.bundle_adjust_with_2frms(frm, ref)
        else:
            frm.cam.R, frm.cam.t, inliers = epnp.ransac_estimate_pose(frm.cam.K, pw, pi)

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
        for i in range(frm_num - 1):
            frm1 = self.frames[i]
            for j in range(i + 1, len(self.frames)):
                frm2 = self.frames[j]
                self.triangulate_2frms(frm1, frm2)

    def update_cam_pose(self):
        for frm in self.frames:
            pw, pi = self.get_epnp_points(frm)
            R, t = epnp.estimate_pose_epnp(frm.cam.K, pw, pi)
            frm.cam.R = R
            frm.cam.t = t

    def plot_map(self, ax):
        for p in self.pw:
            if p is not None:
                p.plot3d(ax, marker='.', color='blue', s=10)

        for frm in self.frames:
            frm.cam.show(ax)
