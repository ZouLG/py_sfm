import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point import *
from camera import *
from geometry import *
import epnp
import map
from frame import Frame
from quarternion import Quarternion
from optimizer import PnpSolver
from utils import set_axis_limit


def plot_data(cameras, pw):
    plt.figure()
    ax = plt.gca(projection='3d')
    for c in cameras:
        c.show(ax)
        c.show_projection(ax, pw)
    plt.pause(0.001)


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


def test_pnp():
    f = 1.05
    pw = generate_sphere_points(16, Point3D((0, 0, 10)), 5)
    # save_points_to_file(pw, "../Data/pw.dat")
    pw = read_points_from_file("../Data/pw.dat")

    camera = PinHoleCamera.place_a_camera((5, 5, 0), (-1, -1, 1), (0, 1, 0), f=f)
    pi, _ = camera.project_world2image(pw)
    pi += np.random.normal(0.0, 6, pi.shape)
    # pi.tofile("../Data/pi.dat")
    pi = np.fromfile("../Data/pi.dat").reshape((-1, 2))
    print("R* = \n", camera.R)
    print("t* = \n", camera.t)
    print(camera.calc_projection_error(pw, pi))
    plot_data([camera], pw)

    # EPNP method
    R, t = epnp.estimate_pose_epnp(camera.K, pw, pi, 4)
    camera = PinHoleCamera(R, t, f=f)
    print("R = \n", R)
    print("t = \n", t)
    print(camera.calc_projection_error(pw, pi))
    plot_data([camera], pw)

    # Non-linear optimization
    q0 = Quarternion.mat_to_quaternion(R) + [0.2, -0.3, 0.5, -0.7]
    t0 = t + [-2.3, 1.5, -3.7]
    solver = PnpSolver([q0, t0], pw, pi, camera.K)
    for i in range(10):
        # solver.solve_t()
        # solver.solve_q()
        solver.solve()
        print(solver.residual)
    camera = PinHoleCamera(Quarternion.quaternion_to_mat(solver.quat), solver.t)
    plot_data([camera], pw)


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


def test_sfm():
    def generate_frms(pi, pc, des, cam):
        des_filter = []
        idx = []
        for i in range(len(des)):
            if not cam.is_out_of_bound(pi[i, :]) and pc[i].z > 0:
                idx.append(i)
                des_filter.append(des[i])
        return Frame(pi[idx, :], des_filter)

    def generate_data(ax):
        f = 1.05
        noise_sigma = 7
        # pw = generate_rand_points(100, [0, 0, 0], [10, 10, 10])
        pw = generate_sphere_points(100, Point3D((0, 0, 0)), 20)
        des = np.random.uniform(0, 128, (len(pw), 128))
        des = des.astype(np.float32)

        # generate frames
        cams, pi, frms = [], [], []
        theta = np.linspace(0.0, np.pi * 1.7, 7)
        center = 15 * np.column_stack((np.zeros(theta.shape), np.cos(theta), np.sin(theta)))
        z_axis = [-center[i, :] for i in range(theta.shape[0])]

        for i in range(theta.shape[0]):
            cams.append(PinHoleCamera.place_a_camera(center[i, :], z_axis[i], (1, 0, 0), f=f))
            pi_, pc_ = cams[i].project_world2image(pw)
            pi_ += np.random.normal(0.0, noise_sigma, pi_.shape)
            frms.append(generate_frms(pi_, pc_, des, cams[i]))
            pi.append(pi_)  # add noise
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
    plt.pause(0.001)

    pt_cloud = map.Map()
    for k, frm in enumerate(frames):
        pt_cloud.add_a_frame(frm)

    ref = pt_cloud.frames[0]
    mat = pt_cloud.frames[1]
    pt_cloud.sort_kps_by_idx()
    pt_cloud.reconstruct_with_2frms(ref, mat, 100)

    plt.figure()
    ax = plt.gca(projection='3d')
    pt_cloud.plot_map(ax)
    set_axis_limit(ax, -20, 20, -10, 30)
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
