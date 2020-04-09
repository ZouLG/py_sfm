import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point import *
from camera import *
from geometry import *
import visualize as vis
import epnp


def generate_cube_points(axis, theta, o=(0., 0., 0.), edge_width=1):
    R = rodriguez(axis, theta)
    ex = np.array((1., 0., 0.))
    ey = np.array((0., 1., 1.))
    ex = np.matmul(R, ex)
    ey = np.matmul(R, ey)
    ez = np.cross(ex, ey)
    edge_half = edge_width / 2.0
    p0 = Point3D(o) + (ex * edge_half - ey * edge_half - ez * edge_half)
    p1 = p0 + ex * edge_width
    p2 = p1 + ey * edge_width
    p3 = p2 - ex * edge_width
    p4 = p0 + ez * edge_width
    p5 = p4 + ex * edge_width
    p6 = p5 + ey * edge_width
    p7 = p6 - ex * edge_width
    return [p0, p1, p2, p3, p4, p5, p6, p7]


def generate_rand_points(num=1, loc=[0, 0, 0], scale=[1, 1, 1]):
    x = np.random.normal(loc[0], scale[0], (num,))
    y = np.random.normal(loc[1], scale[1], (num,))
    z = np.random.normal(loc[2], scale[2], (num,))
    points = []
    for i in range(num):
        points.append(Point3D((x[i], y[i], z[i])))
    return points


def generate_training_data():
    axis = [-1.0, -1.0, 1.0]
    theta = np.pi * 0.6
    R = rodriguez(axis, theta)
    t = np.array([5, 5, 10])
    print("R = \n", R)
    print("t = \n", t)
    plt.figure()
    ax = plt.gca(projection='3d')

    # two cameras with different views
    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera(R, t)
    camera1.show(ax)
    camera2.show(ax)
    print("E = \n", camera2.get_essential_mat())

    p = generate_rand_points(20, [0, 0, 9], [3, 3, 3])
    # p = generate_cube_points((1., 1., 1.), 0 * np.pi / 2, Point3D((0., 0., 25)), 10)
    # p = [Point3D((3, 6, 10))]
    p2d1, p3d1 = camera1.project_world2image(p)
    p2d2, p3d2 = camera2.project_world2image(p)

    camera1.show_projection(ax, p)
    camera2.show_projection(ax, p)
    img1 = camera1.show_projected_img(p)
    img2 = camera2.show_projected_img(p)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    p2d1.tofile("../Data/p2d1.dat")
    p2d2.tofile("../Data/p2d2.dat")
    save_points_to_file(p3d1, "../Data/p3d1.dat")
    save_points_to_file(p3d2, "../Data/p3d2.dat")
    save_points_to_file(p, "../Data/pw.dat")

    lim_low = -5
    lim_high = 10
    ax.set_xlim([lim_low, lim_high])
    ax.set_ylim([lim_low, lim_high])
    ax.set_zlim([lim_low, lim_high])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def check_camera_position():
    plt.figure()
    ax = plt.gca(projection='3d')

    data_type = np.float64
    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    R = np.fromfile("../Data/R.dat", data_type).reshape((3, 3))
    t = np.fromfile("../Data/t.dat", data_type)
    print(R)
    print(t)

    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera(R, t)
    camera1.show(ax)
    camera2.show(ax)

    p3d, p3d1, p3d2 = camera_triangulation(camera1, camera2, p2d1, p2d2)
    c1 = camera1.get_camera_center()
    c2 = camera2.get_camera_center()
    for i in range(len(p3d)):
        # p3d[i].plot3d(ax, color='red', s=5)
        p3d1[i].plot3d(ax, color='red', s=5)
        p3d2[i].plot3d(ax, color='blue', s=5)
        ax.plot3D([c1.x, p3d1[i].x], [c1.y, p3d1[i].y], [c1.z, p3d1[i].z], linestyle='--', color='red', linewidth=1)
        ax.plot3D([c2.x, p3d2[i].x], [c2.y, p3d2[i].y], [c2.z, p3d2[i].z], linestyle='--', color='blue', linewidth=1)

    lim_low = -5
    lim_high = 10
    ax.set_xlim([lim_low, lim_high])
    ax.set_ylim([lim_low, lim_high])
    ax.set_zlim([lim_low, lim_high])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def test_visualizer():
    plt.figure()
    ax = plt.gca(projection='3d')

    R = np.eye(3)
    t = np.array((-1, -1, 0))
    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera(R, t)
    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    vis.test_visualizer(ax, camera1, camera2, p2d1, p2d2)


def test_pnp():
    pi = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    # pi += np.random.normal(0.0, 5, pi.shape)
    pc = read_points_from_file("../Data/p3d2.dat")
    pw = read_points_from_file("../Data/pw.dat")
    print(list2mat(pw))
    camera = PinHoleCamera()
    epnp.estimate_pose_epnp(camera.K, pw, pi, 4)


def test_back_end():
    f = 1.05
    pi1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    pi2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    pi1 += np.random.normal(0.0, 15, pi1.shape)
    pi2 += np.random.normal(0.0, 15, pi2.shape)
    # pw = read_points_from_file("../Data/pw.dat")
    # print(list2mat(pw))

    camera1 = PinHoleCamera(f=f)
    pc1 = camera1.project_image2camera(pi1)
    pc2 = camera1.project_image2camera(pi2)
    E = get_null_space_ransac(list2mat(pc1), list2mat(pc2), max_iter=40)
    R_list, t_list = decompose_essential_mat(E)
    R, t = check_validation_rt(R_list, t_list, pc1, pc2)
    t *= 15

    err = []
    for i in range(30):
        camera2 = PinHoleCamera(f=f, R=R, t=t)
        pw, _, _ = camera_triangulation(camera1, camera2, pi1, pi2)
        err.append(camera1.calc_projection_error(pw, pi1) + camera2.calc_projection_error(pw, pi2))
        R, t = epnp.estimate_pose_epnp(camera2.K, pw, pi2, 4)

    t = t / np.linalg.norm(t) * 15
    print(err)
    plt.figure()
    plt.stem(err)
    print(R)
    print(t)
    R.tofile("../Data/R.dat")
    t.tofile("../Data/t.dat")


def plot_data():
    plt.figure()
    ax = plt.gca(projection='3d')
    data_type = np.float64
    R = np.fromfile("../Data/R.dat", data_type).reshape((3, 3))
    t = np.fromfile("../Data/t.dat", data_type)
    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    p3d1 = read_points_from_file("../Data/p3d1.dat")
    p3d2 = read_points_from_file("../Data/p3d2.dat")
    pw = read_points_from_file("../Data/pw.dat")

    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera(R, t)
    camera1.show(ax)
    camera1.show_projection(ax, pw)
    plt.show()


if __name__ == "__main__":
    # plot_data()
    generate_training_data()
    test_back_end()
    check_camera_position()
    # test_visualizer()
    plt.show()
