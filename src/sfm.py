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
            if k < 5:
                self.map.add_a_frame(Frame(), img, 4)
        frm1 = self.map.frames[0]
        frm2 = self.map.frames[2]

        self.map.sort_kps_in_frame()
        self.map.init_with_2frames(frm1, frm2)

        self.map.sort_kps()
        self.ba.solve_lm()

        for frm in self.map.frames:
            if frm.status is True:
                continue
            self.map.localization(frm)
            self.map.reconstruction(frm)
            self.map.sort_kps()
            self.ba.solve_lm()

        self.map.plot_map(ax)
        set_axis_limit(ax, -70, 70, 30, 170)


if __name__ == "__main__":
    sfm = Sfm(r"F:\zoulugeng\program\python\01.SLAM\Data\data_qinghuamen\image data")
    sfm.reconstruct()
    plt.show()