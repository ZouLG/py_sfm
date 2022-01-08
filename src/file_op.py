import random
import numpy as np
import matplotlib.pyplot as plt
from point import Point3D, list2mat


def dump_object(obj, file):
    import pickle
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_object(file):
    import pickle
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj


def set_axis_limit(ax, center, width):
    assert len(center) == 3
    half = width / 2
    xlow = center[0] - half
    ylow = center[1] - half
    zlow = center[2] - half
    xhigh = center[0] + half
    yhigh = center[1] + half
    zhigh = center[2] + half
    ax.set_xlim([xlow, xhigh])
    ax.set_ylim([ylow, yhigh])
    ax.set_zlim([zlow, zhigh])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_map(pt_list, cam_list, sample=1.0):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    xmin, xmax = float("inf"), -float("inf")
    ymin, ymax = float("inf"), -float("inf")
    zmin, zmax = float("inf"), -float("inf")
    for pt in pt_list:
        if random.uniform(0, 1) > sample:
            continue
        xmin = min(xmin, pt.x)
        xmax = max(xmax, pt.x)
        ymin = min(ymin, pt.y)
        ymax = max(ymax, pt.y)
        zmin = min(zmin, pt.z)
        zmax = max(zmax, pt.z)
        pt.plot3d(ax, marker='.', color='blue', s=1)

    for cam in cam_list:
        cam.show(ax)

    w_x = xmax - xmin
    w_y = ymax - ymin
    w_z = zmax - zmin
    c = (xmin + w_x / 2, ymin + w_y / 2, zmin + w_z / 2)
    w = max(w_x, w_y, w_z)
    set_axis_limit(ax, c, w)


def save_map_to_ply(global_map, file):
    pw = [p for p in global_map.pw if isinstance(p, Point3D)]
    cams = [frm.cam for frm in global_map.frames if frm.status is True]
    pts_num = len(pw)
    cam_num = len(cams)
    pt_cams = []
    for cam in cams:
        pt_cams += cam.get_img_plane()
        pt_cams.append(cam.get_camera_center())

    ply_head = [
        "ply",
        "format ascii 1.0",
        "comment Created by Python Sfm",
        "element vertex %d" % (pts_num + len(pt_cams)),
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element edge %d" % (cam_num * 8),
        "property int vertex1",
        "property int vertex2",
        "element face %d" % 0,
        "end_header"
    ]

    with open(file, 'w') as f:
        for s in ply_head:
            f.writelines(s + "\n")

        """ vertexes of camera center & plane """
        for p in pt_cams:
            f.writelines("%f\t%f\t%f\t" % (p.x, p.y, p.z))
            f.writelines("%d\t%d\t%d\n" % (255, 0, 0))

        """ world points """
        for p in pw:
            f.writelines("%f\t%f\t%f\t" % (p.x, p.y, p.z))
            color = p.color or (0, 0, 255)
            f.writelines("%d\t%d\t%d\n" % (color[0], color[1], color[2]))

        """ edges of camera plane """
        for i in range(cam_num):
            for j in range(4):
                f.writelines("%d\t%d\n" % (5 * i + j, 5 * i + 4))
                f.writelines("%d\t%d\n" % (5 * i + j, 5 * i + (j + 1) % 4))
        print("saved %d points and %d cameras to %s" % (pts_num, cam_num, file))


def test_plot_map():
    pw = [Point3D((0, 0, 0)), Point3D((0, 0, 10)), Point3D((20, 0, 0))]
    plot_map(pw, [])
    plt.show()


def test_save_ply():
    file_name = "../data/test.ply"
    save_to_ply([Point3D((0, 0, 0)), Point3D((0, 1, 0))], file_name)


if __name__ == "__main__":
    test_plot_map()
