from map import Map
from frame import Frame
from point import *
import glob
from sparse_bundler import SparseBa
from utils import save_to_ply


class Sfm(object):
    def __init__(self, dir):
        self.img_name_list = glob.glob(dir + '/*.jpg')
        self.map = Map()
        self.ba = SparseBa(self.map)

    def reconstruct(self):
        plt.figure()
        ax = plt.gca(projection='3d')
        for k, img in enumerate(self.img_name_list):
            if k in [2, 3, 4, 5]:
                self.map.add_a_frame(Frame(), img, 1)

        self.map.sort_kps_in_frame()
        self.map.initialize(k=5)
        self.map.sort_kps()
        save_to_ply(self.map.pw, "../data/pcd_init.ply")
        self.ba.solve_lm()
        save_to_ply(self.map.pw, "../data/pcd_refine.ply")

        Frame.draw_common_kps(self.map.frames[0], self.map.frames[1])
        # Frame.draw_kps(self.map.frames[0].img_data, self.map.frames[0].pi)
        # Frame.draw_kps(self.map.frames[1].img_data, self.map.frames[1].pi)
        plt.pause(0.5)

        while True:
            status, frm = self.map.localise_a_frame()
            if status is True:
                self.ba.solve_lm()
                self.map.reconstruction(frm)
                self.map.sort_kps()
                self.ba.solve_lm()
                save_to_ply(self.map.pw, "../data/pcd_%d.ply" % frm.frm_idx)
                print("frame %d added. %d frames and %d points reconstructed" %
                      (frm.frm_idx, self.map.fixed_frm_num, self.map.fixed_pt_num))
            else:
                break

        self.map.plot_map(ax)


if __name__ == "__main__":
    sfm = Sfm("../data/data_qinghuamen/image data/")
    # sfm = Sfm("../data/GustavIIAdolf/")
    sfm.reconstruct()
    plt.show()
