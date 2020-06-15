from map import Map
from frame import Frame
from point import *
import glob
from sparse_bundler import SparseBa
from utils import set_axis_limit


class Sfm(object):
    def __init__(self, dir):
        self.img_name_list = glob.glob(dir + '/*.jpg')
        self.map = Map()
        self.ba = SparseBa(self.map)

    def reconstruct(self):
        plt.figure()
        ax = plt.gca(projection='3d')
        for k, img in enumerate(self.img_name_list):
            if k < 2:
                self.map.add_a_frame(Frame(), img, 4)
        ref = self.map.frames[0]
        mat = self.map.frames[1]

        self.map.sort_kps_by_idx()
        self.map.reconstruct_with_2frms(ref, mat, 100)

        self.ba.calc_jacobian_mat()

        self.map.plot_map(ax)


if __name__ == "__main__":
    sfm = Sfm(r"F:\zoulugeng\program\python\01.SLAM\Data\data_qinghuamen\image data")
    sfm.reconstruct()
    plt.show()