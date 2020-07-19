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
        self.pj_err_th = 20
        self.total_err = 0
        self.detector = cv2.xfeatures2d_SIFT.create()
        self.scale = 100
        self.fixed_pt_num = 0
        self.fixed_frm_num = 0

    def collect_pnp_points(self, frm):
        idx_in_frm, pw = [], []
        for i, idx in enumerate(frm.kps_idx):
            if idx is not np.Inf and self.pw[idx] is not None:
                idx_in_frm.append(i)
                pw.append(self.pw[idx])
        return pw, frm.pi[idx_in_frm, :], idx_in_frm

    def localise_a_frame(self):
        frm, pw_view, pi_view = self.frames[0], [], None
        for f in self.frames:
            if f.status is False:
                pw, pi, _ = self.collect_pnp_points(f)
                if len(pw) > len(pw_view):
                    frm = f
                    pw_view = pw
                    pi_view = pi

        if len(pw_view) < 15:
            print("Warning: There is no frame has enough points in view")
            return False, None
        frm.cam.R, frm.cam.t, _ = epnp.solve_pnp_ransac(frm.cam.K, pw_view, pi_view, iter=300)
        err = frm.cam.calc_projection_error(pw_view, pi_view)
        if err < self.pj_err_th:
            print("frame %d: %d points in view, re-projection error: %.4f" % (frm.frm_idx, len(pw_view), err))
            frm.pj_err = err
            frm.status = True
            self.fixed_frm_num += 1
            return True, frm
        else:
            print("Warning: frm %d has too few viewed points, pj_err = %.4f" % (frm.frm_idx, err))
            return False, None

    def __is_valid_point__(self, pw, cam1, cam2, _pi1, _pi2, threshold=1):
        pc1 = cam1.project_world2camera([pw])
        pc2 = cam2.project_world2camera([pw])
        if pc1[0].z <= cam1.f or pc2[0].z <= cam2.f:
            return False, np.Inf
        pi1 = cam1.project_camera2image(pc1)
        pi2 = cam2.project_camera2image(pc2)
        err1 = np.linalg.norm(pi1 - _pi1)
        err2 = np.linalg.norm(pi2 - _pi2)
        if err1 > threshold or err2 > threshold:
            return False, err1 + err2
        return True, err1 + err2

    def reconstruction(self, frm):
        if frm.status is False:
            return
        for i, m in enumerate(self.match_map[frm.frm_idx]):
            ref = self.frames[i]
            if m < 100 or ref.status is False:
                continue
            pi0, pi1, idx = self.get_corresponding_matches(ref, frm)
            pw_valid, idx_valid, err = self.reconstruct_with_2frames(frm, ref, pi0, pi1, idx)
            print("reconstruct %d points with frame %d & %d" % (len(pw_valid), ref.frm_idx, frm.frm_idx))
            self.register_points(pw_valid, idx_valid)

    def get_corresponding_matches(self, ref, mat):
        idx0, idx1, idx2 = [], [], []
        for i, k in enumerate(mat.kps_idx):
            if k is np.Inf:
                continue
            j = binary_search(ref.kps_idx, k)
            if j >= 0:
                idx0.append(k)
                idx1.append(j)
                idx2.append(i)
        pi0 = ref.pi[idx1]
        pi1 = mat.pi[idx2]
        return pi0, pi1, idx0

    def estimate_pose_with_2frames(self, ref, mat):
        print("Estimating pose with 2 frames...")
        if ref.status is not True:
            ref.cam.R = np.eye(3)
            ref.cam.t = np.zeros((3,))
        pi0, pi1, idx = self.get_corresponding_matches(ref, mat)
        if len(idx) < 16:
            return None, None, []
        pc0 = ref.cam.project_image2camera(pi0)
        pc1 = mat.cam.project_image2camera(pi1)
        E, _ = get_null_space_ransac(list2mat(pc0), list2mat(pc1), eps=1e-5, max_iter=500)
        R_list, t_list = decompose_essential_mat(E)
        R, t = check_validation_rt(R_list, t_list, pc0, pc1)

        mat.cam.R = np.matmul(ref.cam.R, R)     # R2 = R1 * R
        mat.cam.t = t + np.matmul(np.matmul(mat.cam.R, ref.cam.R.T), ref.cam.t)     # t2 = t + R2 * R1.T * t1
        mat.cam.t = mat.cam.t / np.linalg.norm(mat.cam.t) * self.scale
        return pi0, pi1, idx

    def reconstruct_with_2frames(self, ref, mat, pi0, pi1, kps_idx):
        pw, _, _ = camera_triangulation(ref.cam, mat.cam, pi0, pi1)
        pw_valid, idx_valid, rpj_err = [], [], 0
        for k, p in enumerate(kps_idx):
            if self.pw[p] is None:
                status, err = self.__is_valid_point__(pw[k], ref.cam, mat.cam, pi0[k], pi1[k], threshold=10)
                if status is False:
                    continue
                pw_valid.append(pw[k])
                idx_valid.append(p)
                rpj_err += err
        return pw_valid, idx_valid, rpj_err / len(pw_valid)

    def initialize(self, k=3):
        def find_k_most(k, matches, match_num, match, num):
            if len(match_num) < k:
                match_num.append(num)
                matches.append(match)
            elif match_num[-1] < num:
                match_num[-1] = num
                matches[-1] = match
            else:
                return

            cur = -1
            for i in range(-2, -len(match_num), -1):
                if match_num[i] < num:
                    swap(match_num, i, cur)
                    swap(matches, i, cur)
                    cur = i
                else:
                    break

        frm_num = len(self.frames)
        if frm_num < 2:
            print("Error: Need at least 2 image from different views")
            raise Exception

        matches, match_num = [], []
        for i in range(frm_num):
            for j in range(i + 1, frm_num):
                find_k_most(k, matches, match_num, (i, j), self.match_map[i][j])

        pw_best, idx_best, match_best, err_min = [], [], (0, 1), np.Inf
        ref_pose = (np.eye(3), np.zeros((3, )))
        mat_pose = (np.eye(3), np.zeros((3, )))
        for m in matches:
            ref, mat = self.frames[m[0]], self.frames[m[1]]
            pi0, pi1, idx = self.estimate_pose_with_2frames(ref, mat)
            pw, idx, err = self.reconstruct_with_2frames(ref, mat, pi0, pi1, idx)
            if err < err_min:
                match_best = m
                ref_pose = (ref.cam.R, ref.cam.t)
                mat_pose = (mat.cam.R, mat.cam.t)
                pw_best = pw
                idx_best = idx
                err_min = err
        ref = self.frames[match_best[0]]
        mat = self.frames[match_best[1]]
        if err_min < self.pj_err_th:
            print("Initialize with frame %d & %d, re-projection error = %.5f" % (ref.frm_idx, mat.frm_idx, err_min))
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
            if self.pw[i] is None:
                self.fixed_pt_num += 1
            self.pw[i] = p

    def register_frame(self, frm):
        frm.status = True
        self.fixed_frm_num += 1

    # def reconstruct_with_2frames(self, ref, mat):
    #     print("Estimating pose with 2 frames...")
    #     pi0, pi1, idx = self.get_corresponding_matches(ref, mat)
    #     pc0 = ref.cam.project_image2camera(pi0)
    #     pc1 = mat.cam.project_image2camera(pi1)
    #     E, _ = get_null_space_ransac(list2mat(pc0), list2mat(pc1), eps=1e-6, max_iter=700)
    #     R_list, t_list = decompose_essential_mat(E)
    #     R, t = check_validation_rt(R_list, t_list, pc0, pc1)
    #
    #     mat.cam.R = np.matmul(ref.cam.R, R)
    #     mat.cam.t = np.matmul(ref.cam.R, t)
    #     mat.cam.t = mat.cam.t / np.linalg.norm(mat.cam.t) * self.scale
    #     pw, _, _ = camera_triangulation(ref.cam, mat.cam, pi0, pi1)
    #     ref_err = ref.cam.calc_projection_error(pw, pi0)
    #     mat_err = mat.cam.calc_projection_error(pw, pi1)
    #
    #     print("projection error = %f, %f" % (ref_err, mat_err))
    #     for k, i in enumerate(idx):
    #         if self.pw[i] is None:
    #             self.fixed_pt_num += 1
    #         self.pw[i] = pw[k]
    #     ref.pj_err = ref_err
    #     mat.pj_err = mat_err
    #     ref.status = True
    #     mat.status = True
    #     self.fixed_frm_num += 2

    def add_a_frame(self, frm, *args):
        # detect & match kps of the frm with the map
        f, fx, fy, img_w, img_h = 1.0, 500, 500, 1920, 1080
        if len(args) > 1:
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
                # frm.img_data = color    # for debug use
            elif isinstance(args[0], np.ndarray):
                gray = args[0]
                img_w, img_h = gray.shape[1], gray.shape[0]
                fx, fy = [img_w, img_h] / 2
            frm.pi, frm.des = frm.detect_kps(gray, self.detector)

        frm.kps_idx = [np.Inf] * len(frm.des)
        frm.cam = PinHoleCamera(f=f, fx=fx, fy=fy, img_w=img_w, img_h=img_h)

        cur_idx = len(self.frames)
        frm.frm_idx = cur_idx
        self.match_map.append([])
        print("total %d key-points detected in frame %d" % (len(frm.des), cur_idx))
        if cur_idx == 0:   # if is the first frame
            self.frames.append(frm)
            self.match_map[-1].append(0)
            return

        # match added frame with all old frames to register each key point
        num = len(self.pw)
        for i, ref in enumerate(self.frames):
            idx0, idx1 = Frame.flann_match_kps(np.array(ref.des), np.array(frm.des))
            pi0, pi1 = ref.pi[idx0], frm.pi[idx1]
            # inliers = list(range(len(idx0)))
            _, _, inliers = Frame.ransac_estimate_pose(pi0, pi1, ref.cam, frm.cam)
            print("matching number between (%d, %d) is %d" % (ref.frm_idx, frm.frm_idx, len(inliers)))
            if len(inliers) > 100:  # enough matching pairs
                for k in inliers:
                    if ref.kps_idx[idx0[k]] is np.Inf and frm.kps_idx[idx1[k]] is np.Inf:
                        ref.kps_idx[idx0[k]] = num
                        frm.kps_idx[idx1[k]] = num
                        self.pw.append(None)
                        num += 1
                    elif ref.kps_idx[idx0[k]] is not np.Inf and frm.kps_idx[idx1[k]] is np.Inf:
                        frm.kps_idx[idx1[k]] = ref.kps_idx[idx0[k]]
                    elif frm.kps_idx[idx1[k]] is not np.Inf and ref.kps_idx[idx0[k]] is np.Inf:
                        ref.kps_idx[idx0[k]] = frm.kps_idx[idx1[k]]
                    else:
                        pass
            else:
                inliers = []
            self.match_map[cur_idx].append(len(inliers))
            self.match_map[i].append(len(inliers))
        self.match_map[cur_idx].append(0)   # match_map[i, i] = 0
        self.frames.append(frm)

    def sort_kps_in_frame(self):
        for frm in self.frames:
            frm.sort_kps_by_idx()

    def sort_kps(self):
        idx = list(range(len(self.pw)))
        m, n = 0, len(self.pw) - 1
        while m < n:
            if self.pw[m] is not None:
                m += 1
                continue
            if self.pw[n] is None:
                n -= 1
                continue

            idx[n] = m
            idx[m] = n
            tmp = self.pw[m]
            self.pw[m] = self.pw[n]
            self.pw[n] = tmp
            m += 1
            n -= 1

        for frm in self.frames:
            for i, kp in enumerate(frm.kps_idx):
                if kp is not np.Inf:
                    frm.kps_idx[i] = idx[kp]
            frm.sort_kps_by_idx()

    def update_attribute_dim(self):
        self.fixed_frm_num = 0
        for frm in self.frames:
            if frm.status is True:
                self.fixed_frm_num += 1

        self.fixed_pt_num = 0
        for p in self.pw:
            if p is not None:
                self.fixed_pt_num += 1

    def get_variables(self):
        variables = []
        for frm in self.frames:
            if frm.status is True:
                variables.append(frm.cam.q.q)
                variables.append(frm.cam.t)
        for p in self.pw:
            if p is not None:
                variables.append(p.p)
        return np.concatenate(variables).copy()

    def set_variables(self, var):
        k = 0
        for frm in self.frames:
            if frm.status is True:
                frm.cam.q = Quarternion(var[k: k + 4])
                frm.cam.t = var[k + 4: k + 7]
                k += 7

        for p in self.pw:
            if p is not None:
                p.p = var[k: k + 3]
                k += 3

    def plot_map(self, ax, r=500):
        points = [p for p in self.pw if p is not None]
        for frm in self.frames:
            if frm.status is True:
                points.append(frm.cam.get_camera_center())

        for p in points:
            p.plot3d(ax, marker='.', color='blue', s=0.5)

        for frm in self.frames:
            if frm.status is True:
                frm.cam.show(ax)

        data = list2mat(points)
        center = np.median(data, axis=0)
        ax.set_xlim([center[0] - r, center[0] + r])
        ax.set_ylim([center[1] - r, center[1] + r])
        ax.set_zlim([center[2] - r, center[2] + r])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
