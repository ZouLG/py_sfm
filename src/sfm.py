import glob
from frame import Frame, draw_common_kps
from global_map import GlobalMap
from sparse_bundler import SparseBa
from file_op import save_to_ply
import matplotlib.pyplot as plt


class Sfm(object):
    def __init__(self, dir_name):
        self.img_name_list = glob.glob(dir_name + '/*.jpg')
        self.global_map = GlobalMap()
        self.ba = SparseBa(self.global_map)

    def reconstruct(self):
        for k, img in enumerate(self.img_name_list):
            if k in [0, 1, 2, 3, 4, 5]:
                self.global_map.add_a_frame(Frame(), img, 1)

        # init with 2 frames
        self.global_map.initialize(k=5)
        save_to_ply(self.global_map.pw, "../data/init_sfm.ply")

        self.ba.solve(filter_err=False)
        save_to_ply(self.global_map.pw, "../data/init_ba.ply")

        status = True
        while status:
            status, frm = self.global_map.localise_a_frame()
            if status is True:
                self.ba.solve(filter_err=False)
                save_to_ply(self.global_map.pw, "../data/pcd_%d_before.ply" % frm.frm_idx)

                self.global_map.reconstruction(frm)
                save_to_ply(self.global_map.pw, "../data/pcd_%d_reconstruct.ply" % frm.frm_idx)
                self.ba.solve(filter_err=True)
                save_to_ply(self.global_map.pw, "../data/pcd_%d_after.ply" % frm.frm_idx)

        # self.global_map.plot_map()
        num = 0
        for i, p in enumerate(self.global_map.pw):
            if len(self.global_map.viewed_frames[i]) == 1:
                num += 1
        print(num)

        frm0 = self.global_map.frames[0]
        frm1 = self.global_map.frames[1]
        frm2 = self.global_map.frames[2]
        draw_common_kps(frm0, frm1)
        draw_common_kps(frm1, frm2)
        draw_common_kps(frm0, frm2)


if __name__ == "__main__":
    # sfm = Sfm("../data/data_qinghuamen/image data/")
    sfm = Sfm("../data/GustavIIAdolf/")
    # sfm = Sfm("../data/Cathedral")
    sfm.reconstruct()
    plt.show()
