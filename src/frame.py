from camera import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from quarternion import Quarternion
from utils import *
import exifread
import math


class Frame:
    def __init__(self, pi=None, des=[], cam=PinHoleCamera()):
        """
            pi: 2d image coordinates of matched key points
            des: description of detected key points
            cam: PinHoleCamera model of this frame
        """
        self.pi = pi
        self.des = des
        self.cam = cam

        self.kps_idx = [None] * len(des)
        self.pj_err = np.Inf
        self.frm_idx = None
        self.status = False
        self.img_data = None    # for debug use

    @staticmethod
    def detect_kps(img, detector, response_th=0.0, num_per_blk=7):
        def insert_kp(arr, kp, idx):
            if len(arr) < num_per_blk:
                arr.append(idx)
            elif kps_all[arr[-1]].response < kp.response:
                arr[-1] = idx
            else:
                return

            for k in range(-1, -len(arr), -1):
                if kps_all[arr[k - 1]].response < kp.response:
                    swap(arr, k, k - 1)
                else:
                    break

        kps_all, des_all = detector.detectAndCompute(img, None)
        height, width = img.shape
        blk_h, blk_w = 64, 64
        m, n = int(math.ceil(width / blk_w)), int(math.ceil(height / blk_w))
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
    def draw_kps(img, pi, radius=5, thickness=2):
        draw_img = img.copy()
        for i in range(pi.shape[0]):
            color = tuple(np.random.randint(0, 255, (3,)).tolist())
            cv2.circle(draw_img, tuple(pi[i, :].astype(int)),
                       radius=radius, color=color, thickness=thickness)
        plt.figure()
        plt.imshow(draw_img, cmap='gray')

    @staticmethod
    def draw_common_kps(frm1, frm2, radius=5):
        idx = 0
        img1, img2 = frm1.img_data.copy(), frm2.img_data.copy()
        for m, p in enumerate(frm1.kps_idx):
            n = binary_search(frm2.kps_idx, p)
            if n != -1 and p is not np.Inf:
                idx += 1
                color = tuple(np.random.randint(0, 255, (3,)).tolist())
                cv2.circle(img1, tuple(frm1.pi[m, :].astype(int)),
                           radius=radius, color=color, thickness=5)
                cv2.circle(img2, tuple(frm2.pi[n, :].astype(int)),
                           radius=radius, color=color, thickness=5)
        print("common kps num = %d" % idx)
        plt.figure()
        plt.imshow(img1)
        plt.figure()
        plt.imshow(img2)

    def draw_re_project_error(self, pw_map):
        img = self.img_data.copy()
        error = []
        for i, k in enumerate(self.kps_idx):
            if k is not np.Inf and pw_map[k] is not None:
                pi, pc = self.cam.project_world2image([pw_map[k]])
                pi_ = self.pi[i]
                color = tuple(np.random.randint(0, 255, (3,)).tolist())
                cv2.circle(img, tuple(pi.reshape((-1, )).astype(int)),
                           radius=12, color=color, thickness=3)
                cv2.circle(img, tuple(pi_.astype(int)),
                           radius=7, color=color, thickness=-1)
                error.append(np.linalg.norm(pi - pi_))
        plt.figure()
        plt.imshow(img)
        plt.figure()
        error.sort()
        plt.stem(error)
        return error

    @staticmethod
    def flann_match_kps(des1, des2, knn_ratio=0.8):
        """
            match current frame's kps with pts in the point cloud
            kps_list: kps of the match frame
        """
        if len(des1) == 0 or len(des2) == 0:
            return [], []
        # index_params = dict(algorithm=0, trees=5)
        # search_params = dict(checks=50)
        # matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        pair = {}
        for match in matches:
            if match[0].distance < match[1].distance * knn_ratio:
                pair[match[0].queryIdx] = match[0].trainIdx

        matches = matcher.knnMatch(des2, des1, k=2)
        idx0, idx1 = [], []
        for match in matches:
            if match[0].distance < match[1].distance * knn_ratio:
                if match[0].trainIdx in pair and pair[match[0].trainIdx] == match[0].queryIdx:
                    idx0.append(match[0].trainIdx)
                    idx1.append(match[0].queryIdx)
        return idx0, idx1

    @staticmethod
    def flann_match_kps_old(des1, des2, knn_ratio=0.5):
        """
            match current frame's kps with pts in the point cloud
            kps_list: kps of the match frame
        """
        if len(des1) == 0 or len(des2) == 0:
            return [], []
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        idx0, idx1 = [], []
        for match in matches:
            if match[0].distance < match[1].distance * knn_ratio:
                idx0.append(match[0].queryIdx)
                idx1.append(match[0].trainIdx)
        return idx0, idx1

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
        return [f, fx, fy, img_w, img_h]

    def sort_kps_by_idx(self):
        """
            sort pi & des by kps_idx
        """
        idx = np.array(self.kps_idx).argsort()
        self.pi = self.pi[idx, :]
        self.des = list(np.array(self.des)[idx])
        self.kps_idx = sorted(self.kps_idx)

    @staticmethod
    def ransac_estimate_pose(pi1, pi2, cam1, cam2):
        assert pi1.shape == pi2.shape
        pc1 = cam1.project_image2camera(pi1)
        pc2 = cam2.project_image2camera(pi2)
        try:
            E, inliers = get_null_space_ransac(list2mat(pc1), list2mat(pc2), eps=1e-5, max_iter=500)
        except:
            print("Warning: there are not enough matching points")
            return None, None, []
        R_list, t_list = decompose_essential_mat(E)
        R, t = check_validation_rt(R_list, t_list, pc1, pc2)
        return R, t, inliers


def test_matcher():
    imPath1 = r"..\Data\data_qinghuamen\image data\IMG_5589.jpg"
    imPath2 = r"..\Data\data_qinghuamen\image data\IMG_5590.jpg"
    color1 = cv2.imread(imPath1)
    color2 = cv2.imread(imPath2)
    imshape = color1.shape
    scale = 4
    gray1 = cv2.resize(cv2.cvtColor(color1, cv2.COLOR_RGB2GRAY), (imshape[1] // scale, imshape[0] // scale))
    gray2 = cv2.resize(cv2.cvtColor(color2, cv2.COLOR_RGB2GRAY), (imshape[1] // scale, imshape[0] // scale))
    sift = cv2.xfeatures2d_SIFT.create()

    _, des1, kps1 = Frame.detect_kps(gray1, sift)
    _, des2, kps2 = Frame.detect_kps(gray2, sift)

    print(len(kps1), len(kps2))
    matches = Frame.flann_match_kps(des1, des2)
    print(len(matches[0]), len(matches[1]))

    kps_new1 = [kps1[i] for i in matches[0]]
    kps_new2 = [kps2[i] for i in matches[1]]
    Frame.draw_kps(gray1, kps_new1)
    Frame.draw_kps(gray2, kps_new2)
    plt.show()
    return matches


if __name__ == "__main__":
    # matches = test_matcher()
    # jpg_path = r"../data/image data/IMG_5589.jpg"
    jpg_path = r"../data/GustavIIAdolf/DSC_0351.JPG"
    exif_data = Frame.get_exif_info(jpg_path)
    print(exif_data)
    # img_data = cv2.imread(jpg_path)
    # print(img_data.shape)
