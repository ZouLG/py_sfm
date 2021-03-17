import random
import numpy as np
import matplotlib.pyplot as plt
from point import Point3D, list2mat


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


def save_to_ply(pw, file, scale=1, filter_radius=np.Inf):
    pw = [p for p in pw if isinstance(p, Point3D)]
    data = list2mat(pw)
    center = np.median(data, axis=0)
    pw_filter = []
    for p in data:
        if np.linalg.norm(p - center) > filter_radius:
            continue
        pw_filter.append((p - center) * scale)
    data = np.row_stack(pw_filter)

    ply_head = ["ply", "format ascii 1.0", "comment Created by Python Sfm",
                "element vertex %d" % len(data), "property float x",
                "property float y", "property float z", "end_header"]
    with open(file, 'w') as f:
        for s in ply_head:
            f.writelines(s + "\n")
        for p in data:
            f.writelines("%f %f %f\n" % (p[0], p[1], p[2]))
    print("save %d points to %s" % (data.shape[0], file))


def test_plot_map():
    pw = [Point3D((0, 0, 0)), Point3D((0, 0, 10)), Point3D((20, 0, 0))]
    plot_map(pw, [])
    plt.show()


def test_save_ply():
    file_name = "../data/test.ply"
    save_to_ply([Point3D((0, 0, 0)), Point3D((0, 1, 0))], file_name)


if __name__ == "__main__":
    test_plot_map()
