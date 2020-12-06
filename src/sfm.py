import glob
from frame import Frame
from global_map import GlobalMap
from sparse_bundler import SparseBa
from file_op import save_to_ply
import matplotlib.pyplot as plt


class Sfm(object):
    def __init__(self, dir):
        self.img_name_list = glob.glob(dir + '/*.jpg')
        self.global_map = GlobalMap()
        self.ba = SparseBa(self.global_map)

    def reconstruct(self):
        for k, img in enumerate(self.img_name_list):
            if k in [0, 1, 2]:
                self.global_map.add_a_frame(Frame(), img, 1)

        # init with 2 frames
        self.global_map.initialize(k=5)
        save_to_ply(self.global_map.pw, "../data/init_sfm.ply")

        self.ba.solve_lm()
        save_to_ply(self.global_map.pw, "../data/init_ba.ply")

        status = True
        while status:
            status, frm = self.global_map.localise_a_frame()
            if status is True:
                self.ba.solve_lm()

                self.global_map.reconstruction(frm)
                self.ba.solve_lm()

                points = []
                for i, p in enumerate(self.global_map.pw):
                    if p is not None:
                        points.append(p)
                save_to_ply(points, "../data/pcd_%d_before.ply" % frm.frm_idx)

        # self.global_map.plot_map()
        for f in self.global_map.frames:
            if f.status is True:
                f.draw_re_project_error(self.global_map.pw)


if __name__ == "__main__":
    # sfm = Sfm("../data/data_qinghuamen/image data/")
    sfm = Sfm("../data/GustavIIAdolf/")
    # sfm = Sfm("../data/Cathedral")
    sfm.reconstruct()
    plt.show()
