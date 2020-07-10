from map import Map
from frame import Frame
from point import *
import glob
from sparse_bundler import SparseBa
from utils import set_axis_limit, save_to_ply


class Sfm(object):
    def __init__(self, dir):
        self.img_name_list = glob.glob(dir + '/*.jpg')
        self.map = Map()
        self.ba = SparseBa(self.map)

    def reconstruct(self):
        plt.figure()
        ax = plt.gca(projection='3d')
        for k, img in enumerate(self.img_name_list):
            if k < 3:
                self.map.add_a_frame(Frame(), img, 2)
        frm1 = self.map.frames[0]
        frm2 = self.map.frames[2]

        self.map.sort_kps_in_frame()
        self.map.reconstruct_with_2frames(frm1, frm2)

        self.map.sort_kps()
        self.ba.solve_lm()
        save_to_ply(self.map.pw, "../data/pcd_init.ply")

        for frm in self.map.frames:
            if frm.status is True:
                continue
            print("locating frame %d..." % frm.frm_idx)
            self.map.localization(frm)
            if frm.status is not True:
                continue
            else:
                self.ba.solve_lm()

            print("reconstructing frame %d..." % frm.frm_idx)
            self.map.reconstruction(frm)
            self.map.sort_kps()
            self.ba.solve_lm()

            save_to_ply(self.map.pw, "../data/pcd_%d.ply" % frm.frm_idx)
            print("%d frames located" % self.map.fixed_frm_num)
            print("%d points reconstructed" % self.map.fixed_pt_num)

        self.map.plot_map(ax)


if __name__ == "__main__":
    sfm = Sfm("../data/data_qinghuamen/image data/")
    # sfm = Sfm("../data/GustavIIAdolf/")
    sfm.reconstruct()
    plt.show()
