import cv2
import numpy as np
from point import list2mat
from quarternion import Quarternion
from camera import *     # noqa
from frame import get_common_points, Frame
from utils import save_k_most
import epnp
from queue import deque


class Config(object):
    def __init__(self):
        # kps detection param
        self.knn_ratio = 0.9
        self.double_check = True

        # essential matrix param
        self.e_threshold = 5e-5
        self.ransac_iter = 300

        # pnp params
        self.pnp_pts_num = 50
        self.least_match_num = 20
        self.pnp_pj_th = 50

        # param for filtering error points
        self.pj_err_th = 20
        self.filter_th = 5

        self.scale = 100
        self.window_size = 1e9


class LocalMap(object):
    def __init__(self, global_map, window):
        super(LocalMap, self).__init__()
        # remove frames which status is False
        window = [global_map.frames[i].frm_idx for i in window if global_map.frames[i].status is True]
        self.window = set(window)
        self.pw_index = []
        self.global_map = global_map
        self.split_map_from_global()

    def split_map_from_global(self):
        self.pw_index = []
        for i, p in enumerate(self.global_map.pw):
            if p is None:
                continue
            viewed_num = 0
            for j in self.global_map.viewed_frames[i]:
                if j in self.window:
                    viewed_num += 1
            if viewed_num >= 2:
                self.pw_index.append(i)

    def get_variables(self):
        variables = []
        for frm_idx in self.window:
            frm = self.global_map.frames[frm_idx]
            assert frm.status is True, "Frame %d pose is unknown!" % frm.frm_idx
            variables.append(frm.cam.q.q)
            variables.append(frm.cam.t)
            # variables.append(np.array((frm.cam.fx, frm.cam.fy)))

        for pt_idx in self.pw_index:
            p = self.global_map.pw[pt_idx]
            assert p is not None, "Point3D %d is unknown" % pt_idx
            variables.append(p.p)
        return np.concatenate(variables).copy()

    def set_variables(self, var):
        k = 0
        for frm_idx in self.window:
            frm = self.global_map.frames[frm_idx]
            assert frm.status is True, "Frame %d pose is unknown!" % frm.frm_idx
            frm.cam.q = Quarternion(var[k: k + 4])
            frm.cam.t = var[k + 4: k + 7]
            # frm.cam.fx = var[k + 7]
            # frm.cam.fy = var[k + 8]
            k += 7

        for pt_idx in self.pw_index:
            p = self.global_map.pw[pt_idx]
            assert p is not None, "Point3D %d is unknown" % pt_idx
            p.p = var[k: k + 3]
            k += 3

    def filter_error_points(self):
        pt_num, landmark_num = 0, 0
        for p in self.pw_index:
            pt = self.global_map.pw[p]
            for f in self.global_map.viewed_frames[p].copy():
                frm = self.global_map.frames[f]
                if frm.status is False:
                    continue
                err = frm.calc_projection_error(p, pt)
                # remove the landmark from this frame when projection error is big
                if err > self.global_map.config.filter_th:
                    landmark_num += 1
                    self.global_map.remove_point(p, f)
                    if len(self.global_map.viewed_frames[p]) < 2:
                        self.global_map.viewed_frames[p].clear()
                        self.global_map.pw[p] = None
                        pt_num += 1
                        break
        print("remove %d landmarks, %d points" % (landmark_num, pt_num))


class GlobalMap(object):
    """
        pw: the list of 3d points in world-frame
        frames: list of frames
        window: a queue of frame indexes of key frames
        match_map: 
        viewed_frames:
    """
    def __init__(self, config=Config(), sequential=False, debug=True):
        super(GlobalMap, self).__init__()
        self.pw = []
        self.frames = []
        self.window = deque()

        self.match_map = {}
        self.viewed_frames = []
        self.config = config
        self.detector = cv2.xfeatures2d_SIFT.create()
        self.debug = debug
        self.sequential = sequential

    def add_a_frame(self, frm, *args):
        # detect & match kps of frm with previous frames
        f, fx, fy, img_w, img_h = 1.0, 500, 500, 1920, 1080
        if len(args) > 1:
            if isinstance(args[0], str):
                resize_scale = 1
                if len(args) > 1:
                    resize_scale = args[1]
                color = cv2.imread(args[0])
                exif_info = Frame.get_exif_info(args[0])
                f = exif_info['f']
                fx = exif_info['fx'] / resize_scale
                fy = exif_info['fy'] / resize_scale
                img_h, img_w, _ = color.shape
                img_w = img_w // resize_scale
                img_h = img_h // resize_scale
                color = cv2.resize(color, (img_w, img_h))
                gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                if self.debug:
                    frm.img_data = color    # for debug use
            elif isinstance(args[0], np.ndarray):
                gray = args[0]
                img_w, img_h = gray.shape[1], gray.shape[0]
                fx, fy = img_w / 2, img_h / 2
            else:
                raise TypeError
            frm.pi, frm.des = frm.detect_kps(gray, self.detector)
        exif_info = {"f": f, "fx": fx, "fy": fy, "img_w": img_w, "img_h": img_h}
        frm.cam = PinHoleCamera(**exif_info)

        cur_idx = len(self.frames)
        frm.frm_idx = cur_idx
        self.match_map[cur_idx] = {}
        print("frame %d: exif = %s; %d key-points detected" % (cur_idx, exif_info, len(frm.des)))
        if cur_idx == 0:   # the first Frame
            self.window.append(cur_idx)
            self.frames.append(frm)
            return

        # match added Frame with key frames in sliding window
        num = len(self.pw)
        matching_num = 0
        for ref_idx in self.window:
            ref = self.frames[ref_idx]
            idx0, idx1 = Frame.flann_match_kps(
                des1=np.asarray(ref.des),
                des2=np.asarray(frm.des),
                knn_ratio=self.config.knn_ratio,
                double_check=self.config.double_check
            )

            pi0, pi1 = np.asarray(ref.pi[idx0]), np.asarray(frm.pi[idx1])
            E, inliers = ransac_estimate_pose(
                pi_ref=pi0,
                pi_mat=pi1,
                cam_ref=ref.cam,
                cam_mat=frm.cam,
                eps=self.config.e_threshold,
                max_iter=self.config.ransac_iter,
                solve_pose=False
            )
            print("%d matches between frame (%d, %d)" % (len(inliers), ref.frm_idx, frm.frm_idx))

            if len(inliers) > self.config.least_match_num:  # enough matching pairs
                matching_num += 1
                for k in inliers:
                    if idx0[k] not in ref.pi_pw:    # add a new point
                        if idx0[k] in ref.pi_pw or idx1[k] in frm.pi_pw:
                            continue
                        ref.pi_pw[idx0[k]] = num
                        frm.pi_pw[idx1[k]] = num
                        ref.pw_pi[num] = idx0[k]
                        frm.pw_pi[num] = idx1[k]
                        # create a new point
                        self.pw.append(None)
                        self.viewed_frames.append(set())
                        self.viewed_frames[-1].add(ref.frm_idx)
                        self.viewed_frames[-1].add(frm.frm_idx)
                        num += 1
                    else:   # an existing point
                        pt_idx = ref.pi_pw[idx0[k]]
                        if pt_idx in frm.pw_pi or idx1[k] in frm.pi_pw:
                            continue
                        frm.pw_pi[pt_idx] = idx1[k]
                        frm.pi_pw[idx1[k]] = pt_idx
                        self.viewed_frames[pt_idx].add(frm.frm_idx)
            else:
                inliers = []
            self.match_map[cur_idx][ref_idx] = [len(inliers), E.T]
            self.match_map[ref_idx][cur_idx] = [len(inliers), E]

        self.frames.append(frm)
        if matching_num > 0 or self.sequential is False:
            if len(self.window) >= self.config.window_size:
                self.window.popleft()
            self.window.append(cur_idx)

    def collect_pnp_points(self, frm):
        pi_idx, pw = [], []
        for idx in frm.pw_pi.keys():
            if self.pw[idx] is not None:
                pi_idx.append(frm.pw_pi[idx])
                pw.append(self.pw[idx])
        return pw, frm.pi[pi_idx, :], pi_idx

    def localise_a_frame(self):
        frm_idx, max_num = None, 0
        for f in self.window:
            frm = self.frames[f]
            if frm.status is False:
                num = len([p for p in frm.pw_pi if self.pw[p] is not None])
                if num > max_num:
                    max_num = num
                    frm_idx = f

        if frm_idx is None:
            print("no more frames can be localized")
            return False, None

        frm = self.frames[frm_idx]
        pw, pi, _ = self.collect_pnp_points(frm)
        if len(pw) < self.config.pnp_pts_num:
            print("Warning: Frame %d doesn't has enough points in view" % len(pw))
            return False, None
        frm.cam.R, frm.cam.t, _ = epnp.solve_pnp_ransac(frm.cam.K, pw, pi, iteration=300)
        err = frm.cam.calc_projection_error(pw, pi)
        if err < self.config.pnp_pj_th:
            print("Frame %d: %d points in view, re-projection error: %.4f" % (frm_idx, len(pw), err))
            frm.pj_err = err
            frm.status = True
            return True, frm
        else:
            print("Warning: frm %d has too few viewed points, pj_err = %.4f" % (frm_idx, err))
            return False, None

    def __is_valid_point__(self, pw, cam1, cam2, _pi1, _pi2, threshold=1):
        pc1 = cam1.project_world2camera([pw])
        pc2 = cam2.project_world2camera([pw])
        if pc1[0].z <= cam1.f or pc2[0].z <= cam2.f:
            return False, np.Inf

        # c1 = cam1.get_camera_center()
        # c2 = cam2.get_camera_center()
        # n1 = c1 - pw
        # n2 = c2 - pw
        # theta = np.matmul(n1, n2) / np.linalg.norm(n1) / np.linalg.norm(n2)
        # if np.abs(theta) > 0.9996:  # 0.9965
        #     return False, np.Inf

        pi1 = cam1.project_camera2image(pc1)
        pi2 = cam2.project_camera2image(pc2)
        err1 = np.linalg.norm(pi1 - _pi1)
        err2 = np.linalg.norm(pi2 - _pi2)
        if err1 > threshold or err2 > threshold:
            return False, err1 + err2
        return True, err1 + err2

    def filter_point(self, frm):
        for i, k in enumerate(frm.kps_idx):
            if k is not np.Inf and self.pw[i] is not None:
                pi, pc = frm.cam.project_world2image([self.pw[i]])
                pi_ = frm.pi[i]
                err = np.linalg.norm(pi - pi_)
                if err > 5 or pc[0].z < 1:
                    frm.kps_idx[i] = np.Inf

    def remove_point(self, pt_idx, frm_idx):
        frm = self.frames[frm_idx]
        pi_idx = frm.pw_pi[pt_idx]
        frm.pw_pi.pop(pt_idx)
        frm.pi_pw.pop(pi_idx)
        self.viewed_frames[pt_idx].remove(frm_idx)
        for f in self.viewed_frames[pt_idx]:
            if f in self.match_map[frm_idx]:
                self.match_map[frm_idx][f][0] -= 1
                self.match_map[f][frm_idx][0] -= 1

    def reconstruction(self, frm):
        if frm.status is False:
            return

        matches = self.match_map[frm.frm_idx]
        for k in matches.keys():
            ref = self.frames[k]
            if matches[k][0] < self.config.least_match_num or ref.status is False:
                continue
            pi0, pi1, idx = get_common_points(ref, frm)
            pw_valid, idx_valid, err = self.triangulate_with_2frames(frm.cam, ref.cam, pi0, pi1, idx)
            print("reconstruct %d points with Frame %d & %d" % (len(pw_valid), ref.frm_idx, frm.frm_idx))
            self.register_points(pw_valid, idx_valid)

    def estimate_pose_with_2frames(self, ref, mat):
        # init ref pose if it is unknown
        if ref.status is not True:
            ref.cam.R = np.eye(3)
            ref.cam.t = np.zeros((3,))
        m = self.match_map[ref.frm_idx].get(mat.frm_idx, None)
        if m is None or m[0] < self.config.least_match_num:
            return None, None, []

        pi0, pi1, pw_idx = get_common_points(ref, mat)
        pc0 = ref.cam.project_image2camera(pi0)
        pc1 = mat.cam.project_image2camera(pi1)
        Rs, ts = decompose_essential_mat(m[1])
        R, t = check_validation_rt(Rs, ts, pc0, pc1)

        # mat.cam.R = np.matmul(ref.cam.R, R)     # R2 = R1 * R
        # mat.cam.t = t + np.matmul(np.matmul(mat.cam.R, ref.cam.R.T), ref.cam.t)     # t2 = t + R2 * R1.T * t1
        mat.cam.R = np.matmul(R, ref.cam.R)     # R2 = R * R1
        mat.cam.t = t + np.matmul(R, ref.cam.t)
        mat.cam.t = mat.cam.t / np.linalg.norm(mat.cam.t) * self.config.scale
        return pi0, pi1, pw_idx

    def triangulate_with_2frames(self, ref_cam, mat_cam, pi0, pi1, kps_idx):
        pw, _, _ = camera_triangulation(ref_cam, mat_cam, pi0, pi1)
        pw_valid, idx_valid, rpj_err = [], [], 0
        for k, p in enumerate(kps_idx):
            if self.pw[p] is None:
                status, err = self.__is_valid_point__(
                    pw[k], ref_cam, mat_cam, pi0[k], pi1[k],
                    threshold=np.Inf    # self.config.pj_err_th
                )
                if status is False:
                    continue
                pw_valid.append(pw[k])
                idx_valid.append(p)
                rpj_err += err
        return pw_valid, idx_valid, rpj_err / (len(pw_valid) or 1)

    def initialize(self, k=3):
        frm_num = len(self.window)
        if frm_num < 2:
            print("Error: Need at least 2 image from different views")
            raise Exception

        matches = []
        for i, ii in enumerate(self.window):
            for j in range(0, i):
                jj = self.window[j]
                m = self.match_map[ii].get(jj, None)
                if m is None:
                    continue
                save_k_most(k, matches, ((ii, jj), m), cmp=lambda x, y: x[1][0] > y[1][0])

        pw_best, idx_best, match_best, err_min = [], [], (0, 1), np.Inf
        ref_pose = (np.eye(3), np.zeros((3, )))
        mat_pose = (np.eye(3), np.zeros((3, )))
        for pair, m in matches:
            ref, mat = self.frames[pair[0]], self.frames[pair[1]]
            pi0, pi1, pw_idx = self.estimate_pose_with_2frames(ref, mat)
            if len(pw_idx) < self.config.least_match_num:
                continue
            pw, pw_idx, err = self.triangulate_with_2frames(ref.cam, mat.cam, pi0, pi1, pw_idx)
            print("initialize with %d points in frame %d & %d, err = %.4f" % 
            (len(pw_idx), ref.frm_idx, mat.frm_idx, err))
            if err < err_min:
                match_best = pair
                ref_pose = (ref.cam.R, ref.cam.t)
                mat_pose = (mat.cam.R, mat.cam.t)
                pw_best = pw
                idx_best = pw_idx
                err_min = err
        ref = self.frames[match_best[0]]
        mat = self.frames[match_best[1]]
        if err_min < self.config.pj_err_th:
            print("best initialization is frame %d & %d, re-projection error = %.5f" % 
            (ref.frm_idx, mat.frm_idx, err_min))
            ref.cam.R, ref.cam.t = ref_pose
            mat.cam.R, mat.cam.t = mat_pose
            self.register_frame(ref)
            self.register_frame(mat)
            self.register_points(pw_best, idx_best)
        else:
            print("Error: failed to find 2 frames for initializing")
            raise Exception

    def register_points(self, pw, idx):
        for p, i in zip(pw, idx):
            self.pw[i] = p

    def register_frame(self, frm):
        frm.status = True

    def sort_kps_in_frame(self):
        for frm_idx in self.window:
            self.frames[frm_idx].sort_kps_by_idx()

    def plot_map(self):
        from file_op import plot_map
        points = [p for p in self.pw if p is not None]
        cameras = [f.cam for f in self.frames if f.status is True]
        plot_map(points, cameras)
