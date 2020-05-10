import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point import *
from camera import *
from geometry import *
import visualize as vis
import epnp
import map
from frame import Frame


def set_axis_limit(ax, low, high, zlow=-10, zhigh=10):
    ax.set_xlim([low, high])
    ax.set_ylim([low, high])
    ax.set_zlim([zlow, zhigh])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

def generate_sphere_points(num, o=Point3D((0, 0, 0)), r=1):
    alpha = np.random.uniform(0.0, np.pi * 2, (num,))
    theta = np.random.uniform(0.0, np.pi * 2, (num,))
    plist = []
    for i in range(num):
        x = r * np.cos(theta[i]) * np.cos(alpha[i])
        y = r * np.cos(theta[i]) * np.sin(alpha[i])
        z = r * np.sin(theta[i])
        plist.append(o + (x, y, z))
    return plist


def generate_rand_points(num=1, loc=[0, 0, 0], scale=[1, 1, 1]):
    x = np.random.normal(loc[0], scale[0], (num,))
    y = np.random.normal(loc[1], scale[1], (num,))
    z = np.random.normal(loc[2], scale[2], (num,))
    points = []
    for i in range(num):
        points.append(Point3D((x[i], y[i], z[i])))
    return points


def generate_training_data():
    plt.figure()
    ax = plt.gca(projection='3d')

    # two cameras with different views
    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera.place_a_camera((0, 0, 20), (0, 0, -1), (0, 1, 0))
    print("R = \n", camera2.R)
    print("t = \n", camera2.t)
    camera1.show(ax)
    camera2.show(ax)
    print("E = \n", camera2.get_essential_mat())

    p = generate_rand_points(20, [0, 0, 9], [5, 5, 5])
    # p = generate_sphere_points(25, Point3D((0, 0, 10)), 5)
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

    set_axis_limit(ax, -10, 10)


def check_camera_position():
    ax = plt.gca(projection='3d')
    data_type = np.float64
    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    R = np.fromfile("../Data/R.dat", data_type).reshape((3, 3))
    t = np.fromfile("../Data/t.dat", data_type)

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

    set_axis_limit(ax, -10, 10)


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
    f = 1.05
    pi = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    pi += np.random.normal(0.0, 15, pi.shape)
    pw = list2mat(read_points_from_file("../Data/pw.dat"))
    pw += np.random.normal(0.0, 0.0, pw.shape)
    pw = mat2list(pw)

    camera = PinHoleCamera(f=f)
    # epnp.estimate_pose_epnp(camera.K, pw, pi, 4)
    R, t, pw, pi = epnp.ransac_estimate_pose(camera.K, pw, pi, iter=10, threshold=50)
    R.tofile("../Data/R.dat")
    t.tofile("../Data/t.dat")
    print(len(pw))


def test_back_end():
    f = 1.05
    t_norm = 20
    pi1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    pi2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    pi1 += np.random.normal(0.0, 15, pi1.shape)
    pi2 += np.random.normal(0.0, 15, pi2.shape)

    camera1 = PinHoleCamera.place_a_camera((0, 0, 0), (0, 0, 1), (0, 1, 0), f=f)
    # camera1 = PinHoleCamera(f=f)
    pc1 = camera1.project_image2camera(pi1)
    pc2 = camera1.project_image2camera(pi2)
    E, _ = get_null_space_ransac(list2mat(pc1), list2mat(pc2), max_iter=20)
    R_list, t_list = decompose_essential_mat(E)
    R, t = check_validation_rt(R_list, t_list, pc1, pc2)
    R = np.matmul(R, camera1.R)
    t *= t_norm

    err = []
    s = 0.2
    plt.figure()
    for i in range(50):
        camera2 = PinHoleCamera(f=f, R=R, t=t)
        _, pw1, pw2 = camera_triangulation(camera1, camera2, pi1, pi2)
        pw = [p * s + q * (1 - s) for p, q in zip(pw1, pw2)]
        err.append(camera1.calc_projection_error(pw, pi1) + camera2.calc_projection_error(pw, pi2))
        # R, t = epnp.estimate_pose_epnp(camera2.K, pw, pi2, 4)
        R, t, _ = epnp.ransac_estimate_pose(camera2.K, pw, pi2, 10, 10)
        R.tofile("../Data/R.dat")
        (t / np.linalg.norm(t) * t_norm).tofile("../Data/t.dat")
        plt.clf()
        check_camera_position()
        plt.pause(0.001)

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


def test_sfm():
    def generate_frms(pi, des, cam):
        des_filter = []
        idx = []
        for i in range(len(des)):
            if not cam.is_out_of_bound(pi[i, :]):
                idx.append(i)
                des_filter.append(des[i])
        return Frame(pi[idx, :], des_filter)

    def generate_data(ax):
        f = 1.00
        noise_sigma = 8
        # pw = generate_rand_points(100, [0, 0, 0], [10, 10, 10])
        pw = generate_sphere_points(100, Point3D((0, 0, 0)), 20)
        des = np.random.uniform(0, 128, (len(pw), 128))
        des = des.astype(np.float32)

        # generate frames
        cams, pi = [], []
        theta = np.linspace(0.0, np.pi * 1.7, 7)
        center = 15 * np.column_stack((np.zeros(theta.shape), np.cos(theta), np.sin(theta)))
        for i in range(theta.shape[0]):
            cams.append(PinHoleCamera.place_a_camera(center[i, :], -center[i, :], (1, 0, 0), f=f))
            pi_, _ = cams[i].project_world2image(pw)
            pi.append(pi_ + np.random.normal(0.0, noise_sigma, pi_.shape))  # add noise

        frms = []
        for i, c in enumerate(cams):
            frms.append(generate_frms(pi[i], des, cams[i]))
            print(len(frms[i].des))

        save_points_to_file(pw, "../Data/pw.dat")

        # show cameras & key points
        for c in cams:
            c.show(ax)
            # c.show_projection(ax, pw)
        for p in pw:
            p.plot3d(ax, marker='.', s=10)
        set_axis_limit(ax, -20, 20, -10, 20)
        return frms

    # --------- test sfm start ----------
    plt.figure()
    ax = plt.gca(projection='3d')
    frames = generate_data(ax)

    plt.figure()
    ax = plt.gca(projection='3d')
    pt_cloud = map.Map()
    for k, frm in enumerate(frames):
        pt_cloud.add_a_frame(frm)
        for i in range(k * 5):
            pt_cloud.update_points()
            pt_cloud.update_cam_pose()

        plt.cla()
        pt_cloud.plot_map(ax)
        set_axis_limit(ax, -20, 20, -10, 30)
        plt.pause(0.001)

    print("best matches =\n", pt_cloud.best_match)
    plt.figure()
    ax = plt.gca(projection='3d')
    print("start optimization...")
    for i in range(30):
        pt_cloud.update_points()
        pt_cloud.update_cam_pose()
        pt_cloud.calc_projecting_err()
        print(pt_cloud.total_err)
        plt.cla()
        pt_cloud.plot_map(ax)
        set_axis_limit(ax, -20, 20, 0, 40)
        plt.pause(0.001)
    print("sfm task finished")


if __name__ == "__main__":
    # plot_data()
    # generate_training_data()
    # test_pnp()
    test_sfm()
    # test_back_end()
    # check_camera_position()
    # test_visualizer()
    plt.show()
