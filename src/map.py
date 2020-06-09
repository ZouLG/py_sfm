from camera import *
from frame import Frame
from utils import *
import epnp


class Map(object):
    def __init__(self):
        self.pw = []
        self.frames = []    # list of frames

        self.match_map = []
        self.status = False
        self.pj_err_th = 10
        self.total_err = 0
        self.detector = cv2.xfeatures2d_SIFT.create()

    def get_pnp_points(self, frm):
        idx_in_frm, pw = [], []
        for i, idx in enumerate(frm.kps_idx):
            if self.pw[idx] is not None:
                idx_in_frm.append(i)
                pw.append(self.pw[idx])
        return pw, frm.pi[idx_in_frm, :], idx_in_frm

    def localization(self, frm):
        pw, pi, _ = self.get_epnp_points(frm)
        if len(pw) < 8:
            print("Warning: too few key points")
            return
        R, t = epnp.ransac_estimate_pose(frm.cam.K, pw, pi, iter=70)
        cam_tmp = PinHoleCamera(R, t, frm.cam.K)
        err = cam_tmp.calc_projection_error(pw, pi)
        if err < frm.pj_err:
            frm.cam = cam_tmp
            frm.pj_err = err
            frm.status = err < self.pj_err_th

    def reconstruction(self, frm):
        if frm.status is False:
            return
        pw, pi, idx = self.get_epnp_points(frm)
        pw_pj = bundle_projection(frm.cam, pw, pi)
        for k, p in zip(idx, pw_pj):
            self.__slide_mean_update_points(frm.kps_idx[k], p)

    def get_corresponding_matches(self, ref, mat):
        idx0, idx1, idx2 = [], [], []
        for i, k in enumerate(mat.kps_idx):
            j = binary_search(ref.kps_idx, k)
            if j >= 0:
                idx0.append(k)
                idx1.append(j)
                idx2.append(i)
        pi0 = ref.pi[idx1]
        pi1 = mat.pi[idx2]
        return pi0, pi1, idx0

    def select_two_frames(self):
        pass

    def init_with_2frms(self, t_scale, iter=20):
        print("Estimating pose with 2 frames...")
        ref, mat = self.select_two_frames()
        pi0, pi1, idx = self.get_corresponding_matches(ref, mat)
        pc0 = ref.cam.project_image2camera(pi0)
        pc1 = mat.cam.project_image2camera(pi1)
        E, _ = get_null_space_ransac(list2mat(pc0), list2mat(pc1), eps=1e-2, max_iter=70)
        R_list, t_list = decompose_essential_mat(E)
        R, t = check_validation_rt(R_list, t_list, pc0, pc1)
        mat.cam.R = np.matmul(ref.cam.R, R)
        mat.cam.t = np.matmul(ref.cam.R, t)
        mat.cam.t = mat.cam.t / np.linalg.norm(mat.cam.t) * t_scale
        _, pw, _ = camera_triangulation(ref.cam, mat.cam, pi0, pi1)
        err_min = ref.cam.calc_projection_error(pw, pi0) + mat.cam.calc_projection_error(pw, pi1)
        pw_best = pw

        ratio = 0.0
        for i in range(iter):
            R, t = epnp.estimate_pose_epnp(mat.cam.K, pw, pi1)
            t = t / np.linalg.norm(t) * t_scale
            cam_tmp = PinHoleCamera(R, t, f=mat.cam.f, sx=mat.cam.sx, sy=mat.cam.sy,
                                    img_w=mat.cam.img_w, img_h=mat.cam.img_h)
            _, pw0, pw1 = camera_triangulation(ref.cam, cam_tmp, pi0, pi1)
            pw = [p * ratio + q * (1 - ratio) for p, q in zip(pw0, pw1)]
            err1 = ref.cam.calc_projection_error(pw, pi0)
            err2 = cam_tmp.calc_projection_error(pw, pi1)
            err = err1 + err2
            if err < err_min:
                err_min = err
                mat.cam = cam_tmp
                pw_best = pw

        if err_min > self.pj_err_th:
            print("Error: init with 2 frames failed, err = %f" % err_min)
            mat.status = False
            return False
        else:
            print("projection error = %f" % err_min)
            for k, i in enumerate(idx):
                self.pw[i] = pw_best[k]
                self.slide_win_size[i] += 1
            ref.status = True
            mat.status = True
            return True

    def add_a_frame(self, frm, *args):
        # detect & match kps of the frm with the map
        f, fx, fy, img_w, img_h = 1.0, 500, 500, 1920, 1080
        if len(args) > 0:
            if isinstance(args[0], str):
                resize_scale = 1
                if len(args) > 1:
                    resize_scale = args[1]
                f, fx, fy, img_w, img_h = Frame.get_exif_info(args[0])
                img_w = int(img_w / resize_scale)
                img_h = int(img_h / resize_scale)
                fx /= resize_scale
                fy /= resize_scale
                color = cv2.resize(cv2.imread(args[0]), (img_w, img_h))
                gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            elif isinstance(args[0], np.ndarray):
                gray = args[0]
                img_w, img_h = gray.shape[1], gray.shape[0]
                fx, fy = [img_w, img_h] / 2
            frm.pi, frm.des, _ = frm.detect_kps(gray, self.detector)
            frm.kps_idx = [None] * len(frm.des)
        frm.cam = PinHoleCamera(f=f, fx=fx, fy=fy, img_w=img_w, img_h=img_h)
        frm.img = color

        cur_idx = len(self.frames)
        self.match_map.append([])
        if cur_idx == 0:   # if is the first frame
            self.frames.append(frm)
            self.match_map[-1].append(0)
            return

        # match added frame with all old frames to register each key point
        num = len(self.pw)
        for i, ref in enumerate(self.frames):
            idx0, idx1 = Frame.flann_match_kps(np.array(ref.des), np.array(frm.des))
            pi0 = ref.pi[idx0]
            pi1 = frm.pi[idx1]
            _, _, inliers = Frame.ransac_estimate_pose(pi0, pi1, ref.cam, frm.cam)
            for k in inliers:
                if ref.kps_idx[idx0[k]] is None:
                    ref.kps_idx[idx0[k]] = num
                    frm.kps_idx[idx1[k]] = num
                    self.pw.append(None)
                    num += 1
            ref.draw_kps(ref.img)
            frm.draw_kps(frm.img)
            self.match_map[cur_idx].append(len(inliers))
            self.match_map[i].append(len(inliers))

        # sort the kps by its index in the map
        # frm.sort_kps_by_idx()
        self.frames.append(frm)

    def localize_and_reconstruct(self):
        # estimate pose of the frame
        for i, frm in enumerate(self.frames):
            if frm.status is False:
                self.localization(frm)
                self.reconstruction(frm)
                self.refine_map()

        for i, frm1 in enumerate(self.frames):
            if frm1.status is False:
                continue
            for j in range(i + 1, len(self.frames)):
                frm2 = self.frames[j]
                if frm2.status is True:
                    self.triangulate_2frms(frm1, frm2)

    def update_points(self):
        for i, frm in enumerate(self.frames):
            self.reconstruction(frm)

    def update_cam_pose(self):
        for frm in self.frames:
            pw, pi, _ = self.get_epnp_points(frm)
            if len(pw) < 5:
                # frm.status = False
                continue
            frm.cam.R, frm.cam.t = epnp.estimate_pose_epnp(frm.cam.K, pw, pi)
            err = frm.cam.calc_projection_error(pw, pi)
            if err < frm.pj_err:
                frm.status = True
                frm.pj_err = err

    def reset_scale(self, scale):
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
