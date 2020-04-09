from point import *
from camera import *
import frame as frm
import key_point as kp


class PointCloud:
    def __init__(self):
        self.kps_list = []

    def plot_points(self):
        pass


class Map(PointCloud):
    def __init__(self):
        self.frame_list = []    # list of frames
        self.frame_idx = []     # index of frames of each kp

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

    def add_frame(self, frame):
        pass
