import key_point as kp
import camera
from point import *
import cv2
import numpy as np
import geometry as geo
from matplotlib import pyplot as plt


class Frame:
    def __init__(self, cam):
        """
            pi: 2d image coordinates of matched key points
            kps_idx: index of key points in the PointCloud list
            camera: pin-hole camera model
        """
        self.pi = None
        self.cam = cam
        self.kps_idx = []

    @staticmethod
    def detect_kps(img, detector):
        kps, des = detector.detectAndCompute(img, None)
        kps_list = []
        for p, d in zip(kps, des):
            kps_list.append(kp.KeyPoint(p, [d]))
        return kps_list

    @staticmethod
    def draw_kps(img, kps_list, color=(255, 0, 0)):
        draw_img = cv2.drawKeypoints(img, kps_list, img, color=color)
        plt.figure()
        plt.imshow(draw_img)

    @classmethod
    def bf_match_kps(self, pt_cloud, frm_kps, threshold=300):
        """
            match current frame's kps with pts in the point cloud
            pt_cloud: a PointCloud
            kps_list: kps of the match frame
        """
        assert self.pi.shape[0] == len(frm_kps)

        map_size = len(pt_cloud.kps_list)
        new_size = map_size
        pt_cloud.frame_list.append(self)
        for p in frm_kps:
            dis_knn = [np.Inf, np.Inf]
            idx = -1
            for i in range(map_size):
                q = pt_cloud.kps_list[i]
                dis = geo.calc_min_dis(p.des, q.des)
                if dis < dis_knn[0]:
                    dis_knn[1] = dis_knn[0]
                    dis_knn[0] = dis
                    idx = i
                elif dis < dis_knn[1]:
                    dis_knn[1] = dis

            if (dis_knn[0] < threshold) and (dis_knn[0] < dis_knn[1] * 0.5):
                pt_cloud.kps_list[idx].des.append(p.des[0])
                self.kps_idx.append(idx)
            else:   # add this point to the map if there is no existing matching pt in the map
                pt_cloud.kps_list.append(p)
                self.kps_idx.append(new_size)
                new_size += 1



if __name__ == "__main__":
    imPath1 = r"..\Data\data_qinghuamen\image data\IMG_5602.jpg"
    color = cv2.imread(imPath1)
    imshape = color.shape
    scale = 4
    gray = cv2.resize(cv2.cvtColor(color, cv2.COLOR_RGB2GRAY), (imshape[1] // scale, imshape[0] // scale))
    sift = cv2.xfeatures2d_SIFT.create()

    kps_list = Frame.detect_kps(color, sift)
    Frame.draw_kps(color, kps_list)
    plt.show()
    print(len(kps_list))

