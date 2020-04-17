from point import *
from camera import *
import frame as frm
import key_point as kp
import numpy as np


class PointCloud:
    def __init__(self, pw=[]):
        self.kps_list = pw

    def plot_points(self):
        pass


class Map(PointCloud):
    def __init__(self, pw=[]):
        self.kps_list = pw
        self.frame_list = []    # list of frames

    @staticmethod
    def solve_2views(frame1, frame2):
        """
            init map with 2 frames which have most matching key points
            frame1 & frame2: 2 frames
            return kp_list, frame_list
        """
        pass

    def estimate_pose(self, frame):
        """
            estimate pose of the matched frame
            frame: the frame with its
        """
        pass

    def add_frame(self, frm, frm_kps):
        pass


