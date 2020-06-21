from camera import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
from quarternion import Quarternion
from utils import *
import exifread


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
        self.status = False

    @staticmethod
    def detect_kps(img, detector):
        kps, des = detector.detectAndCompute(img, None)
        pi = np.zeros((len(kps), 2))
        for i in range(len(kps)):
            pi[i, :] = kps[i].pt
        return pi, des, kps

    def draw_kps(self, img, radius=5, color=(255, 0, 0)):
        draw_img = img
        for i in range(self.pi.shape[0]):
            if self.kps_idx[i] is not np.Inf:
                cv2.circle(draw_img, (int(self.pi[i, 0]), int(self.pi[i, 1])),
                           radius=radius, color=color, thickness=1)
        # draw_img = cv2.drawKeypoints(img, kps, img, color=color)
        plt.figure()
        plt.imshow(draw_img, cmap='gray')

    @staticmethod
    def flann_match_kps(des1, des2, knn_ratio=0.5):
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
        def get_data_with_tag(exif_data, tag):
            if tag in ['EXIF ExifImageWidth', 'EXIF ExifImageLength']:
                return exif_data[tag].values[0]
            elif tag in ['EXIF FocalLength', 'EXIF FocalPlaneXResolution', 'EXIF FocalPlaneYResolution']:
                ratio = exif_data[tag].values[0]
                return ratio.num / ratio.den
            elif tag in ['EXIF FocalPlaneResolutionUnit']:
                if exif_data[tag].values[0] == 2:
                    return 25.4     # 1 inch = 25.4 mm
                elif exif_data[tag].values[0] == 3:
                    return 10
                elif exif_data[tag].values[0] == 4:
                    return 1
            return None

        fobj = open(jpg_path, 'rb')
        exif_data = exifread.process_file(fobj)
        f = get_data_with_tag(exif_data, 'EXIF FocalLength')
        nx = get_data_with_tag(exif_data, 'EXIF FocalPlaneXResolution')
        ny = get_data_with_tag(exif_data, 'EXIF FocalPlaneYResolution')
        xy_unit = get_data_with_tag(exif_data, 'EXIF FocalPlaneResolutionUnit')
        sx = xy_unit / nx
        sy = xy_unit / ny
        fx = f / sx
        fy = f / sy
        img_w = get_data_with_tag(exif_data, 'EXIF ExifImageWidth')
        img_h = get_data_with_tag(exif_data, 'EXIF ExifImageLength')
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
            E, inliers = get_null_space_ransac(list2mat(pc1), list2mat(pc2), eps=1e-3, max_iter=200)
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
    jpg_path = r"F:\zoulugeng\program\python\01.SLAM\Data\data_qinghuamen\image data\IMG_5589.jpg"
    exif_data = Frame.get_exif_info(jpg_path)
    print(exif_data)
    # img_data = cv2.imread(jpg_path)
    # print(img_data.shape)
