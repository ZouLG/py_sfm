import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point import *
from camera import *
from geometry import *


def generate_cube_points(axis, theta, o=(0., 0., 0.), edge_width=1):
    R = rodriguez(axis, theta)
    ex = np.array((1., 0., 0.))
    ey = np.array((0., 1., 1.))
    ex = np.matmul(R, ex)
    ey = np.matmul(R, ey)
    ez = np.cross(ex, ey)
    p0 = Point3D(o)
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
    axis = [-0.5, 1, 1.]
    theta = np.pi * 0.9
    R = rodriguez(axis, theta)
    t = np.array([0, -3, 6])
    plt.figure()
    ax = plt.gca(projection='3d')

    # two cameras with different views
    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera(R, t)
    print(np.matmul(camera2.T, camera2.T_))
    # camera2.trans_camera(R, t)
    camera1.show(ax)
    camera2.show(ax)

    # p = [Point3D((0, 0, 10))]
    p = generate_rand_points(20, [0, 0, 5], [2, 2, 1])
    # p = generate_cube_points((0., 0., 1.), np.pi / 4, Point3D((0., 0., 5)), 2.5)
    p2d1, p3d1 = camera1.project(p)
    p2d2, p3d2 = camera2.project(p)

    camera1.show_projection(ax, p)
    camera2.show_projection(ax, p)
    p2d1.tofile("../Data/p2d1.dat")
    p2d2.tofile("../Data/p2d2.dat")
    print(p2d1)
    print(p2d2)

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([0, 10])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def read_data():
    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    p2d1 = np.column_stack((p2d1, np.ones((p2d1.shape[0], 1))))
    p2d2 = np.column_stack((p2d2, np.ones((p2d2.shape[0], 1))))
    print(p2d1)
    print(p2d2)
    return p2d1, p2d2


if __name__ == "__main__":
    generate_training_data()
