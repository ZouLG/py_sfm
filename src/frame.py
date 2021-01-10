import cv2
import math
import exifread
import numpy as np
from matplotlib import pyplot as plt
from point import *   # noqa
from camera import *    # noqa
from utils import swap

__all__ = ["get_common_points", "draw_kps", "draw_matched_pts", "Frame"]


def get_common_points(ref, frm):
    i, j = 0, 0
    ref_pw = list(ref.pw_pi.keys())
    frm_pw = list(frm.pw_pi.keys())
    ref_pw.sort()
    frm_pw.sort()
    m, n = len(ref_pw), len(frm_pw)
    pw_idx, idx0, idx1 = [], [], []
    while i < m and j < n:
        k1 = ref_pw[i]
        k2 = frm_pw[j]
        if k1 < k2:
            i += 1
        elif k1 > k2:
            j += 1
        else:
            pw_idx.append(k1)
            idx0.append(ref.pw_pi[k1])
            idx1.append(frm.pw_pi[k2])
            i += 1
            j += 1
    pi0 = ref.pi[idx0]
    pi1 = frm.pi[idx1]
    return pi0, pi1, pw_idx


def draw_kps(img, pi, radius=None, thickness=5):
    if radius is None:
        radius = max(img.shape) // 200
    draw_img = img.copy()
    for i in range(pi.shape[0]):
        color = tuple(np.random.randint(0, 255, (3,)).tolist())
        cv2.circle(draw_img, tuple(pi[i, :].astype(int)),
            radius=radius, color=color, thickness=thickness)
    plt.figure()
    plt.imshow(draw_img)


def draw_common_kps(frm1, frm2, radius=None):
    if radius is None:
        radius = max(frm1.img_data.shape) // 200
    assert frm1.img_data is not None and frm2.img_data is not None
    img1, img2 = frm1.img_data.copy(), frm2.img_data.copy()
    pi1, pi2, pw_idx = get_common_points(frm1, frm2)
    for p, q in zip(pi1, pi2):
        color = tuple(np.random.randint(0, 255, (3,)).tolist())
        cv2.circle(img1, tuple(p.astype(int)), radius=radius, color=color, thickness=5)
        cv2.circle(img2, tuple(q.astype(int)), radius=radius, color=color, thickness=5)
    print("common kps num = %d" % len(pw_idx))
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)


def draw_matched_pts(img1, img2, pts1, pts2, radius=None):
    assert len(pts1) == len(pts2)
    img1 = img1.copy()
    img2 = img2.copy()
    if radius is None:
        radius = max(img1.shape) // 200
    for p, q in zip(pts1, pts2):
        color = tuple(np.random.randint(0, 255, (3,)).tolist())
        cv2.circle(img1, tuple(p.astype(int)),
                   radius=radius, color=color, thickness=5)
        cv2.circle(img2, tuple(q.astype(int)),
                   radius=radius, color=color, thickness=5)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)


class Frame(object):
    def __init__(self, pi=None, des=[], cam=PinHoleCamera()):
        """
            pi: 2d image coordinates of matched key points
            des: description of detected key points
            cam: PinHoleCamera model of this Frame
        """
        super(Frame, self).__init__()
        self.pi = pi
        self.des = des
        self.cam = cam

        self.pw_pi = {}     # key: pw_idx   value: pi_idx
        self.pi_pw = {}     # key: pi_idx   value: pw_idx
        self.pj_err = np.Inf
        self.frm_idx = None
        self.status = False
        self.img_data = None    # for debug use

    def register_point(self, pt_idx, pi_idx):
        self.pw_pi[pt_idx] = pi_idx
        self.pi_pw[pi_idx] = pt_idx

    @staticmethod
    def detect_kps(img, detector, response_th=0.0, num_per_blk=10, blk_size=None):
        def insert_kp(arr, kp, idx):
            if len(arr) < num_per_blk:
                arr.append(idx)
            elif kps_all[arr[-1]].response < kp.response:
                arr[-1] = idx
            else:
                return

            for kk in range(-1, -len(arr), -1):
                if kps_all[arr[kk - 1]].response >= kp.response:
                    break
                swap(arr, kk, kk - 1)

        kps_all, des_all = detector.detectAndCompute(img, None)
        height, width = img.shape
        if blk_size is None:
            blk_h, blk_w = height // 50, width // 50
        else:
            blk_h, blk_w = blk_size

        m, n = int(math.ceil(width / blk_w)), int(math.ceil(height / blk_h))
        kps_idx = []
        for i in range(n):
            kps_idx.append([])
            for j in range(m):
                kps_idx[i].append([])

        pi, des = [], []
        for i, kp in enumerate(kps_all):
            if kp.response < response_th:
                continue
            x = int(kp.pt[0] / blk_w)
            y = int(kp.pt[1] / blk_h)
            insert_kp(kps_idx[y][x], kp, i)

        for i in range(n):
            for j in range(m):
                for k in kps_idx[i][j]:
                    pi.append(np.array(kps_all[k].pt))
                    des.append(des_all[k])
        pi = np.row_stack(pi)
        return pi, des

    @staticmethod
    def flann_match_kps(des1, des2, knn_ratio=0.9, double_check=True):
        if len(des1) == 0 or len(des2) == 0:
            return [], []
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(np.asarray(des1), np.asarray(des2), k=2)
        pair = {}
        for match in matches:
            if match[0].distance < match[1].distance * knn_ratio:
                pair[match[0].queryIdx] = match[0].trainIdx

        if double_check:
            idx0, idx1 = [], []
            matches = matcher.knnMatch(np.asarray(des2), np.asarray(des1), k=2)
            for match in matches:
                if match[0].distance < match[1].distance * knn_ratio:
                    if match[0].trainIdx in pair.keys() and pair[match[0].trainIdx] == match[0].queryIdx:
                        idx0.append(match[0].trainIdx)
                        idx1.append(match[0].queryIdx)
            return idx0, idx1
        else:
            return pair.values(), pair.keys()

    @staticmethod
    def get_exif_info(jpg_path):
        def get_focal_length(exif_data):
            f, f35 = None, None
            if 'EXIF FocalLength' in exif_data.keys():
                ratio = exif_data['EXIF FocalLength'].values[0]
                f = ratio.num / ratio.den

            f35_keys = ['EXIF FocalLengthIn35mmFilm', 'Image FocalLengthIn35mmFilm']
            for key in f35_keys:
                if key in exif_data.keys():
                    f35 = exif_data[key].values[0]
            return f, f35

        def get_sensor_res(exif_data, f, f35, unit, img_w, img_h):
            sx, sy = None, None
            if 'EXIF FocalPlaneXResolution' in exif_data.keys():
                ratio = exif_data['EXIF FocalPlaneXResolution'].values[0]
                nx = ratio.num / ratio.den
                sx = unit / nx
            elif f35 is not None:
                sx = f / f35 * np.sqrt(1872 / (img_w ** 2 + img_h ** 2))

            if 'EXIF FocalPlaneYResolution' in exif_data.keys():
                ratio = exif_data['EXIF FocalPlaneYResolution'].values[0]
                ny = ratio.num / ratio.den
                sy = unit / ny
            elif f35 is not None:
                sy = sx
            return sx, sy

        def get_sensor_res_unit(exif_data):
            units = [None, None, 25.4, 10, 1]   # 2: inch   3: cm   4: mm
            unit_keys = ['EXIF FocalPlaneResolutionUnit', 'Image ResolutionUnit']
            for key in unit_keys:
                if key in exif_data.keys():
                    return units[exif_data[key].values[0]]
            return None

        def get_image_res(exif_data):
            img_w, img_h = None, None
            if 'EXIF ExifImageWidth' in exif_data.keys():
                img_w = exif_data['EXIF ExifImageWidth'].values[0]

            if 'EXIF ExifImageLength' in exif_data.keys():
                img_h = exif_data['EXIF ExifImageLength'].values[0]
            return img_w, img_h

        fobj = open(jpg_path, 'rb')
        exif_data = exifread.process_file(fobj)
        f, f35 = get_focal_length(exif_data)
        img_w, img_h = get_image_res(exif_data)
        xy_unit = get_sensor_res_unit(exif_data)
        sx, sy = get_sensor_res(exif_data, f, f35, xy_unit, img_w, img_w)
        fx = f / sx
        fy = f / sy
        return {"f": f, "fx": fx, "fy": fy, "img_w": img_w, "img_h": img_h}

    def calc_projection_error(self, pt_idx, pt):
        assert pt_idx in self.pw_pi, \
            "point %d is not viewed in frame %d" % (pt_idx, self.frm_idx)
        pi = self.pi[self.pw_pi[pt_idx]]
        err = self.cam.calc_projection_error([pt], pi)
        return err

    def draw_re_project_error(self, point_map, threshold=None):
        img = self.img_data.copy()
        radius = max(img.shape) // 200
        error = []
        for k in self.pw_pi:
            if point_map[k] is not None:
                pi, pc = self.cam.project_world2image([point_map[k]])
                pi_ = self.pi[self.pw_pi[k]]
                if threshold is None:
                    color = tuple(np.random.randint(0, 255, (3,)).tolist())
                elif np.sqrt(np.sum((pi - pi_) ** 2)) > threshold:
                    color = (255, 0, 0)
                else:
                    color = (0, 0, 255)

                cv2.circle(img, tuple(pi.reshape((-1, )).astype(int)),
                           radius=radius // 2, color=color, thickness=-1)
                cv2.circle(img, tuple(pi_.astype(int)),
                           radius=radius, color=color, thickness=1)
                error.append(np.linalg.norm(pi - pi_))
        plt.figure()
        plt.imshow(img)
        plt.figure()
        error.sort()
        plt.stem(error)
        return error


def test_matcher():
    scale = 1
    im_path1 = r"..\Data\data_qinghuamen\image data\IMG_5589.jpg"
    im_path2 = r"..\Data\data_qinghuamen\image data\IMG_5591.jpg"
    # im_path1 = r"..\Data\GustavIIAdolf\DSC_0351.jpg"
    # im_path2 = r"..\Data\GustavIIAdolf\DSC_0352.jpg"
    # im_path1 = r"..\Data\Cathedral\dkyrkan2 001.jpg"
    # im_path2 = r"..\Data\Cathedral\dkyrkan2 004.jpg"
    color1 = cv2.imread(im_path1)
    color2 = cv2.imread(im_path2)
    imshape = color1.shape
    color1 = cv2.resize(color1, (imshape[1] // scale, imshape[0] // scale))
    color2 = cv2.resize(color2, (imshape[1] // scale, imshape[0] // scale))
    gray1 = cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY)

    sift = cv2.xfeatures2d_SIFT.create()
    kps1, des1 = Frame.detect_kps(gray1, sift)
    kps2, des2 = Frame.detect_kps(gray2, sift)
    # draw_kps(color1, kps1)
    # draw_kps(color2, kps2)
    print("frame1: %d kps, frame2: %d kps" % (len(kps1), len(kps2)))

    matches = Frame.flann_match_kps(des1, des2, knn_ratio=0.9, double_check=True)
    print(len(matches[0]), len(matches[1]))
    kps_new1, kps_new2 = kps1[matches[0]], kps2[matches[1]]
    # circle matched kps with same color
    draw_matched_pts(color1, color2, kps_new1, kps_new2)

    exif1 = Frame.get_exif_info(im_path1)
    exif2 = Frame.get_exif_info(im_path2)
    camera1 = PinHoleCamera(**exif1)
    camera2 = PinHoleCamera(**exif2)
    camera2.R, camera2.t, inliers = ransac_estimate_pose(
        kps_new1,
        kps_new2,
        camera1,
        camera2,
        eps=5e-4,
        max_iter=300,
        t_scale=100
    )
    print("%d matched kps after filtered by ransac" % len(inliers))
    kps1, kps2 = kps_new1[inliers], kps_new2[inliers]
    pw, _, _ = camera_triangulation(camera1, camera2, kps1, kps2)
    draw_matched_pts(color1, color2, kps1, kps2)

    print("%d points reconstructed" % len(pw))
    from file_op import plot_map, save_to_ply
    save_to_ply(pw, "../data/init.ply")
    plot_map(pw, [])
    plt.show()


def test_exif():
    jpg_path = r"../data/GustavIIAdolf/DSC_0351.JPG"
    exif_data = Frame.get_exif_info(jpg_path)
    print(exif_data)


if __name__ == "__main__":
    test_exif()
    test_matcher()
