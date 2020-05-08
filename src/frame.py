from camera import *
import cv2
import numpy as np
import epnp
import geometry as geo
from matplotlib import pyplot as plt
from utils import *


class Frame:
    def __init__(self,
                 pi=None,
                 des=[],
                 R=np.eye(3),
                 t=np.zeros((3,)),
                 f=1.0
                 ):
        """
            pi: 2d image coordinates of matched key points
            des: description of detected key points
            kps_idx: index of key points in the Map
            cam: pin-hole camera model
        """
        self.pi = pi
        self.des = des

        self.kps_idx = [None] * len(des)
        self.pw = [None] * len(des)
        self.cam = PinHoleCamera(R, t, f=f)

    @staticmethod
    def detect_kps(img, detector):
        kps, des = detector.detectAndCompute(img, None)
        pi = np.zeros((len(kps), 2))
        for i in range(len(kps)):
            pi[i, :] = kps[i].pt
        return pi, des, kps

    @staticmethod
    def draw_kps(img, kps, color=(255, 0, 0)):
        draw_img = cv2.drawKeypoints(img, kps, img, color=color)
        plt.figure()
        plt.imshow(draw_img)

    @staticmethod
    def flann_match_kps(des1, des2):
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
        idx0 = []
        idx1 = []
        for match in matches:
            if match[0].distance < match[1].distance * 0.7:
                idx0.append(match[0].queryIdx)
                idx1.append(match[0].trainIdx)
        return idx0, idx1

    @staticmethod
    def get_focus_len_by_exif(exif):
        pass

    def sort_kps_by_idx(self):
        """
            sort pi & des by kps_idx
        """
        idx = np.array(self.kps_idx).argsort()
        self.pi = self.pi[idx, :]
        self.des = list(np.array(self.des)[idx])

    @staticmethod
    def ransac_estimate_pose(pi1, pi2, cam1, cam2):
        assert pi1.shape == pi2.shape
        pc1 = cam1.project_image2camera(pi1)
        pc2 = cam2.project_image2camera(pi2)
        try:
            E, inliers = get_null_space_ransac(list2mat(pc1), list2mat(pc2), eps=1e-3, max_iter=15)
        except:
            print("Warning: there are not enough matching points")
            return None, None, []

        R_list, t_list = decompose_essential_mat(E)
        R, t = check_validation_rt(R_list, t_list, pc1, pc2)
        return R, t, inliers

    def estimate_pose_and_points(self, ref, t_scale=15):
        idx0 = []
        idx1 = []
        for i in range(len(self.kps_idx)):
            k = self.kps_idx[i]
            j = binary_search(ref.kps_idx, k)
            if j >= 0:
                idx0.append(j)
                idx1.append(i)
        pi0 = get_point_by_idx(ref.pi, idx0)
        pi1 = get_point_by_idx(self.pi, idx1)
        pc0 = ref.cam.project_image2camera(pi0)
        pc1 = self.cam.project_image2camera(pi1)
        try:
            E, inliers = get_null_space_ransac(list2mat(pc0), list2mat(pc1), eps=1e-3, max_iter=12)
        except:
            print("Warning: there are not enough kp matches")
            return [], []
        R_list, t_list = decompose_essential_mat(E)
        R, t = check_validation_rt(R_list, t_list, pc0, pc1)
        t = t / np.linalg.norm(t) * t_scale
        self.cam.R = np.matmul(ref.cam.R, R)
        self.cam.t = np.matmul(ref.cam.R, t)
        pw, pw1, pw2 = camera_triangulation(ref.cam, self.cam, pi0, pi1)
        return pw, idx1


def test_matcher():
    imPath1 = r"..\Data\data_qinghuamen\image data\IMG_5602.jpg"
    imPath2 = r"..\Data\data_qinghuamen\image data\IMG_5603.jpg"
    color1 = cv2.imread(imPath1)
    color2 = cv2.imread(imPath2)
    imshape = color1.shape
    scale = 8
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
    matches = test_matcher()

