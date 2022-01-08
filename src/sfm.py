import cv2
import glob
from frame import Frame, draw_common_kps
from global_map import GlobalMap
from sparse_bundler import SparseBa
from file_op import save_map_to_ply, dump_object, load_object
import matplotlib.pyplot as plt


class Sfm(object):
    def __init__(self, dir_name):
        self.img_name_list = glob.glob(dir_name + '/*.jpg')
        self.global_map = None
        self.bundler = None
        self.detector = cv2.xfeatures2d_SIFT.create()

    def debug(self):
        frm1 = self.global_map.frames[4]
        frm2 = self.global_map.frames[5]
        draw_common_kps(frm1, frm2)
        plt.show()

    def reconstruct(self, dumped_map_path=None):
        if dumped_map_path is None:
            self.global_map = GlobalMap()
            for k, img in enumerate(self.img_name_list):
                if k < 5:
                    self.global_map.add_a_frame(
                        Frame(),
                        jpg_name=img,
                        resize_scale=2,
                        detector=self.detector
                    )
            dump_object(self.global_map, "../data/global_map.dat")
        else:
            """ load global map from file """
            self.global_map = load_object(dumped_map_path)

        print("total %d frames" % len(self.global_map.frames))

        # self.debug()
        """ sparse bundler optimizer """
        self.bundler = SparseBa(self.global_map)

        """ init with 2 frames """
        self.global_map.initialize(k=5)
        self.bundler.solve(filter_err=True)
        save_map_to_ply(self.global_map, "../data/init_ba.ply")

        status = True
        while status:
            print("\n################ localizing a frame... ################")
            status, frm = self.global_map.localise_a_frame()

            if status is True:
                self.bundler.solve(filter_err=True, window=[frm.frm_idx])
                _, idx = self.global_map.reconstruction(frm)
                self.bundler.solve(filter_err=True, pw_index=idx)

                self.bundler.solve(filter_err=True)
                save_map_to_ply(self.global_map, "../data/bundler_add_frm%d.ply" % frm.frm_idx)
                dump_object(self.global_map, "../data/global_map_add_frm%d.dat" % frm.frm_idx)

        self.global_map.plot_map(sample=0.2)
        fixed_frm_num = len([f for f in self.global_map.frames if f.status is True])
        print("%d frames localized, %d un-localized" % (fixed_frm_num, len(self.global_map.frames) - fixed_frm_num))


if __name__ == "__main__":
    # dataset_path = "../data/data_qinghuamen/image data/"
    # dataset_path = "../data/Cathedral"
    dataset_path = "../data/house/"

    sfm = Sfm(dataset_path)
    sfm.reconstruct()
    # sfm.reconstruct("../data/global_map.dat")
    plt.show()
