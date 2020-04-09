import key_point as kp
import camera
from point import *
import cv2
from matplotlib import pyplot as plt


class Frame:
    def __init__(self, pi):
        """
            pi: 2d image coordinates of matched key points
            kps_idx: index of key points in the PointCloud list
            camera: pin-hole camera model
        """
        self.pi = pi
        self.kps_idx = []
        self.camera = camera.PinHoleCamera()

    @staticmethod
    def detect_kps(img, detector):
        kps, des = detector.detectAndCompute(img, None)
        kps_list = []
        for p, d in zip(kps, des):
            kps_list.append(kp.KeyPoint(p, d))
        return kps_list

    @staticmethod
    def draw_kps(img, kps_list, color=(255, 0, 0)):
        draw_img = cv2.drawKeypoints(img, kps_list, img, color=color)
        plt.figure()
        plt.imshow(draw_img)

    @classmethod
    def match_kps(self, pt_cloud, kps_list, matcher):
        """
            match current frame's kps with pts in the point cloud
            pt_cloud: a PointCloud
            kps_list: kps of the match frame
            matcher: matcher
        """
        pass


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

