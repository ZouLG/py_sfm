import cv2
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
        self.detector = cv2.xfeatures2d_SIFT.create()

    def reconstruct(self):
        for k, img in enumerate(self.img_name_list):
            if k > 0:
                self.global_map.add_a_frame(
                    Frame(),
                    jpg_name=img,
                    resize_scale=2,
                    detector=self.detector
                )

        # init with 2 frames
        self.global_map.initialize(k=5)
        save_to_ply(self.global_map.pw, "../data/init_sfm.ply")

        self.ba.solve(filter_err=True)
        save_to_ply(self.global_map.pw, "../data/init_ba.ply")

        status = True
        while status:
            print("################ localizing a frame ################\n")
            status, frm = self.global_map.localise_a_frame()
            if status is True:
                self.ba.solve(filter_err=True, window=[frm.frm_idx])
                save_to_ply(self.global_map.pw, "../data/pcd_%d_before_ba.ply" % frm.frm_idx)

                _, idx = self.global_map.reconstruction(frm)
                save_to_ply(self.global_map.pw, "../data/pcd_%d_reconstruct.ply" % frm.frm_idx)
                self.ba.solve(filter_err=True, pw_index=idx)
                save_to_ply(self.global_map.pw, "../data/pcd_%d_reconstruct_after_ba.ply" % frm.frm_idx)

                self.ba.solve(filter_err=True)
                save_to_ply(self.global_map.pw, "../data/pcd_%d_after_ba.ply" % frm.frm_idx)

        # self.global_map.plot_map(sample=0.2)
        fixed_frms = len([f for f in self.global_map.frames if f.status is True])
        un_fixed_frms = len(self.global_map.frames) - fixed_frms
        print("%d frames localized, %d un-localized" % (fixed_frms, un_fixed_frms))

        num = 0
        for i, p in enumerate(self.global_map.pw):
            if len(self.global_map.viewed_frames[i]) == 1:
                num += 1
        print(num)

        # frm0 = self.global_map.frames[0]
        # frm1 = self.global_map.frames[1]
        # frm2 = self.global_map.frames[2]
        # test(self.global_map)
        # draw_common_kps(frm0, frm1)
        # draw_common_kps(frm1, frm2)
        # draw_common_kps(frm0, frm2)


if __name__ == "__main__":
    # sfm = Sfm("../data/data_qinghuamen/image data/")
    sfm = Sfm("../data/GustavIIAdolf/")
    # sfm = Sfm("../data/Cathedral")
    sfm.reconstruct()
    plt.show()
