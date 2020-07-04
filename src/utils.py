from point import *


def set_axis_limit(ax, low, high, zlow=-10, zhigh=10):
    ax.set_xlim([low, high])
    ax.set_ylim([low, high])
    ax.set_zlim([zlow, zhigh])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def binary_search(g, k):
    start = 0
    end = len(g) - 1
    while start <= end:
        mid = start + (end - start) // 2
        if g[mid] == k:
            return mid
        elif g[mid] < k:
            start = mid + 1
        else:
            end = mid - 1
    return -1


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
    print(data.shape)

    ply_head = ["ply", "format ascii 1.0", "comment Created by python sfm",
                "element vertex %d" % len(data), "property float x",
                "property float y", "property float z", "end_header"]
    with open(file, 'w') as f:
        for s in ply_head:
            f.writelines(s + "\n")
        for p in data:
            f.writelines("%f %f %f\n" % (p[0], p[1], p[2]))
    print("save %d points" % data.shape[0])


def plot_ply(file):
    pass


if __name__ == "__main__":
    file_name = "../data/test.ply"
    save_to_ply([Point3D((0, 0, 0)), Point3D((0, 1, 0))], file_name)
