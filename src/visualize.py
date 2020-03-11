import msvcrt
import numpy as np
from point import *
from camera import *
import geometry as geo
from mpl_toolkits.mplot3d import Axes3D


class ProjectionVisualizer:
    def __init__(self, ax):
        self.ax = ax
        self.x_lim_low = -1
        self.x_lim_high = 1
        self.y_lim_low = -1
        self.y_lim_high = 1
        self.z_lim_low = -1
        self.z_lim_high = 1
        self.cameras = []

    def set_lim(self, obj):
        if isinstance(obj, list):
            for p in p_list:
                self.x_lim_low  = min(p.x, self.x_lim_low)
                self.x_lim_high = max(p.x, self.x_lim_high)
                self.y_lim_low  = min(p.y, self.y_lim_low)
                self.y_lim_high = max(p.y, self.y_lim_high)
                self.z_lim_low  = min(p.z, self.z_lim_low)
                self.z_lim_high = max(p.z, self.z_lim_high)

    def add_camera(self, camera):
        self.cameras.append(camera)


def plot_ray(ax, p, n, ishalf=True, color='red', linestype='-', linewidth=1, length=30):
    """
    plot a ray in 3D axis
        ax: the 3D axis
        p: the fixed 3D point lies in the ray
        n: the direction of the ray
        half: if True plot a half-ray
    """
    length = 30
    n_ = length * n / np.linalg.norm(n)
    if ishalf:
        p0 = p
    else:
        p0 = p - n_
    p1 = p + n_
    ax.plot3D([p0.x, p1.x], [p0.y, p1.y], [p0.z, p1.z], color=color, linestyle=linestype, linewidth=linewidth)


def test_visualizer(ax, camera1, camera2, kps1=None, kps2=None):
    def set_ax_attribute():
        ax.cla()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim([-15, 5])
        ax.set_ylim([-5, 15])
        ax.set_zlim([-5, 15])

    def plot_camera():
        set_ax_attribute()
        camera1.show(ax)
        camera2.show(ax)

    def plot_back_projection():
        c1 = camera1.get_camera_center()
        c2 = camera2.get_camera_center()
        p3dc1 = camera1.project_image2camera(kps1)
        p3dc2 = camera2.project_image2camera(kps2)
        p3dw1 = camera1.project_camera2world(p3dc1)
        p3dw2 = camera2.project_camera2world(p3dc2)
        colors = ['red', 'blue', 'green', 'purple']
        for i in range(4):
            n1 = p3dw1[i] - c1
            n2 = p3dw2[i] - c2
            p, p1, p2 = triangulation(n1, n2, c1, c2)
            line_type = '-'
            if np.linalg.norm(p2 - p1) < 0.05:
                line_type = '--'
            plot_ray(ax, c1, n1, color=colors[i], linestype=line_type, length=np.linalg.norm(n1))
            plot_ray(ax, c2, n2, color=colors[i], linestype=line_type, length=np.linalg.norm(n2))

    dth = np.pi * 0.01
    dl = 1e-1
    ex = np.array((1, 0, 0)) * dl
    ey = np.array((0, 1, 0)) * dl
    ez = np.array((0, 0, 1)) * dl

    key = 'a'
    while ord(key) != 27:
        plot_camera()
        plot_back_projection()

        plt.draw()
        plt.pause(0.001)
        key = msvcrt.getch()
        if ord(key) == ord('w'):        # shift alone +ez
            print("up")
            camera2.shift_by_t(ez)
        elif ord(key) == ord('s'):      # shift alone -ez
            print("down")
            camera2.shift_by_t(-ez)
        elif ord(key) == ord('d'):      # shift alone +ex
            print("right")
            camera2.shift_by_t(ex)
        elif ord(key) == ord('a'):      # shift alone -ex
            print("left")
            camera2.shift_by_t(-ex)
        elif ord(key) == ord('q'):      # shift alone +ey
            print("front")
            camera2.shift_by_t(ey)
        elif ord(key) == ord('e'):      # shift alone -ey
            print("back")
            camera2.shift_by_t(-ey)
        elif ord(key) == ord('j'):      # rotate around ex by +dth
            print("rotate around ex by dth")
            camera2.rotate_around_axis(dth, axis=0)
        elif ord(key) == ord('u'):      # rotate around ex by -dth
            print("rotate around ex by -dth")
            camera2.rotate_around_axis(-dth, axis=0)
        elif ord(key) == ord('k'):      # rotate around ey by +dth
            print("rotate around ey by dth")
            camera2.rotate_around_axis(dth, axis=1)
        elif ord(key) == ord('i'):      # rotate around ey by -dth
            print("rotate around ey by -dth")
            camera2.rotate_around_axis(-dth, axis=1)
        elif ord(key) == ord('l'):      # rotate around ez by +dth
            print("rotate around ez by dth")
            camera2.rotate_around_axis(dth, axis=2)
        elif ord(key) == ord('o'):      # rotate around ez by -dth
            print("rotate around ez by -dth")
            camera2.rotate_around_axis(-dth, axis=2)
        elif ord(key) == ord('p'):      # plot figure
            plt.pause(2)


if __name__ == "__main__":
    plt.figure()
    ax = plt.gca(projection='3d')

    axis = [1., 0., 1.]
    theta = np.pi / 2
    R = geo.rodriguez(axis, theta)
    t = np.array([-4, -4, 0])

    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera(R, t)
    camera1.show(ax)
    camera2.show(ax)

    camera2.rotate_around_axis(np.pi / 2, 0)
    camera2.show(ax)

    test_visualizer(ax, camera1, camera2)
