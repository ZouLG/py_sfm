import open3d as o3d
import numpy as np


def plot_pcd(file_name, scale=1, filter_radius=np.Inf):
    pcd = o3d.io.read_point_cloud(file_name)
    data = np.asarray(pcd.points)
    center = np.median(data, axis=0)
    pw_filter = []
    for p in data:
        if np.linalg.norm(p - center) > filter_radius:
            continue
        pw_filter.append((p - center) * scale)

    data = np.row_stack(pw_filter)
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    pcd_file = "../data/qinghuamen.ply"
    plot_pcd(pcd_file, scale=1, filter_radius=100)
