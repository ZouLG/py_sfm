import numpy as np
from point import *
import geometry as geo


def back_project(K, p):
    """
    back project 2D image point to 3D space, and normalize the coordinates into homogeneous coordinates
        param K: the intrinsic matrix
        param p: a Mx2 matrix with M 2D points
        return: a list of 3D homogeneous points
    """
    ph = []
    for i in range(p.shape[0]):
        q = np.matmul(K, p[i, :])
        ph.append(Point3D(q / q[2]))
    return ph


def check_validation_rt(R, t, p1, p2):
    """
    eliminate the solutions of R & t decomposed from essential matrix
        R: list of rotation matrix of camera2, length is 2
        t: list of shift vectors of camera2
        p1: list of 3D points of camera1
        p2: list of 3D points of camera2
        return: the R & t satisfy all Z > 0 constrain
    """
    R_ = [np.linalg.pinv(R[0]), np.linalg.pinv(R[1])]
    idx_r = [0, 0, 1, 1]
    idx_t = [0, 1, 0, 1]
    idx = [True] * 4
    for p, q in zip(p1, p2):
        for i in range(len(idx_r)):
            if not idx[i]:
                continue
            m = idx_r[i]
            n = idx_t[i]
            p2 = np.matmul(R[m], p.p) + t[n]      # transpose from world frame to camera2 frame
            if p2[2] <= 0:
                idx[i] = False
                continue
            p1 = np.matmul(R_[m], q.p - t[n])      # transpose from camera2 to world frame
            if p1[2] <= 0:
                idx[i] = False
                continue

        if sum(idx) == 1:
            break
    n = [i for i in range(len(idx)) if idx[i]]
    print("length of candidate R&t is %d, n[0] = %d" % (len(n), n[0]))
    return R[n[0]], t[n[0]]


def update_camera_plane(camera):
    img_center = camera.img_center
    img_w = camera.sy * camera.img_w
    img_h = camera.sx * camera.img_h
    p0 = img_center + camera.ex * img_h / 2 - camera.ey * img_w / 2
    p1 = img_center + camera.ex * img_h / 2 + camera.ey * img_w / 2
    p2 = img_center - camera.ex * img_h / 2 + camera.ey * img_w / 2
    p3 = img_center - camera.ex * img_h / 2 - camera.ey * img_w / 2
    return ImgPlane(p0, p1, p2, p3)


class PinHoleCamera:
    def __init__(self, R=np.eye(3),
                 t=np.array((0., 0., 0.)),
                 f=1., sx=0.002, sy=0.002,
                 img_w=1920, img_h=1080):
        self.T = geo.homo_rotation_mat(R, t)
        self.__dict__['o'] = Point3D((0., 0., 0.))
        self.__dict__['ex'] = np.array((1.0, 0.0, 0.0))
        self.__dict__['ey'] = np.array((0.0, 1.0, 0.0))
        self.__dict__['ez'] = np.cross(self.ex, self.ey)

        # intrinsic params
        self.__dict__['f'] = f
        self.__dict__['sx'] = sx
        self.__dict__['sy'] = sy
        self.__dict__['img_w'] = img_w
        self.__dict__['img_h'] = img_h
        self.__dict__['img_center'] = self.o + self.f * self.ez
        self.__dict__['img'] = update_camera_plane(self)

    def __setattr__(self, key, value):
        if key in ['f', 'sx', 'sy', 'img_w', 'img_h']:
            self.__dict__[key] = value
            self.__dict__['img_center'] = self.o.p + self.f * self.ez
            self.__dict__['img'] = update_camera_plane(self)
        elif key == 'o':
            self.__dict__[key] = Point3D(value)
        elif key in ['R', 't']:
            self.__dict__[key] = value
            self.__dict__['R_'] = np.linalg.pinv(self.R)
            self.__dict__['T'] = geo.homo_rotation_mat(self.R, self.t)
            self.__dict__['T_'] = np.linalg.pinv(self.T)
        elif key == 'T':
            self.__dict__[key] = value
            self.__dict__['R'] = value[0:3, 0:3]
            self.__dict__['t'] = value[0:3, 3]
            self.__dict__['R_'] = np.linalg.pinv(self.R)
            self.__dict__['T_'] = np.linalg.pinv(self.T)
        else:
            print("Caution: Attribute %s can not be set" % (key))

    def trans_camera(self, *args):
        if len(args) == 2:
            R = args[0]
            t = np.array(args[1]).reshape((3,))
            T = geo.homo_rotation_mat(R, t)
            self.T = np.matmul(self.T, T)
        elif len(args) == 1:
            T = args[0]
            self.T = np.matmul(self.T, T)
        else:
            print("Error: trans_camera takes R&t or T as parameters")

    def show(self, ax, color='blue', s=20):
        o = geo.rotate3d(self.o, self.T_)
        o.plot3d(ax, color=color, s=s)
        ex = np.matmul(self.R_, self.ex)
        ey = np.matmul(self.R_, self.ey)
        ez = np.matmul(self.R_, self.ez)
        ax.quiver(o.x, o.y, o.z, ex[0], ex[1], ex[2], normalize=True, color='red')
        ax.quiver(o.x, o.y, o.z, ey[0], ey[1], ey[2], normalize=True, color='green')
        ax.quiver(o.x, o.y, o.z, ez[0], ez[1], ez[2], normalize=True, color='blue')
        p0 = geo.rotate3d(self.img.p0, self.T_)
        p1 = geo.rotate3d(self.img.p1, self.T_)
        p2 = geo.rotate3d(self.img.p2, self.T_)
        p3 = geo.rotate3d(self.img.p3, self.T_)
        img = ImgPlane(p0, p1, p2, p3)
        img.show(ax, color=color)
        ax.plot3D([o.x, img.p0.x], [o.y, img.p0.y], [o.z, img.p0.z], color=color)
        ax.plot3D([o.x, img.p1.x], [o.y, img.p1.y], [o.z, img.p1.z], color=color)
        ax.plot3D([o.x, img.p2.x], [o.y, img.p2.y], [o.z, img.p2.z], color=color)
        ax.plot3D([o.x, img.p3.x], [o.y, img.p3.y], [o.z, img.p3.z], color=color)

    # project a point in the world frame to the image plane, return the img coordinate
    def project(self, plist):
        p2d = []
        p3d = []
        for p in plist:
            p = geo.rotate3d(p, self.T)
            op = p - self.o
            oq = np.matmul(op, self.ez)
            a = self.f / oq
            opi = a * op
            pi = self.o + opi     # Point projected on the image plane
            p0pi = pi - self.img.p0
            i = -np.matmul(p0pi, self.ex) / self.sx
            j = np.matmul(p0pi, self.ey) / self.sy
            if i < 0 or i >= self.img_h or j < 0 or j >= self.img_w:
                print("Warning: point projected out of image")
            p2d.append(np.array([i, j]))
            p3d.append(pi)
        return np.array(p2d), np.array(p3d)     # 2d & 3d coordinate

    def is_out_of_bound(self, pi):
        if pi[0] < 0 or pi[0] >= self.img_h or pi[1] < 0 or pi[1] >= self.img_w:
            return True
        else:
            return False

    def show_projection(self, ax, psrc):
        p2d, p3d = self.project(psrc)
        for i in range(len(psrc)):
            o = geo.rotate3d(self.o, self.T_)       # back project to the world frame
            X = psrc[i]
            p = p3d[i]
            q = p2d[i]
            X.plot3d(ax, s=10, marker='o', color='blue')
            if not self.is_out_of_bound(q):
                ax.plot3D([o.x, X.x], [o.y, X.y], [o.z, X.z], color='blue', linestyle='--', linewidth=1)
                p = geo.rotate3d(p, self.T_)
                p.plot3d(ax, s=10, marker='o', color='red')
            else:
                ax.plot3D([o.x, X.x], [o.y, X.y], [o.z, X.z], color='red', linestyle='--', linewidth=1)


if __name__ == "__main__":
    plt.figure()
    ax = plt.gca(projection='3d')
    p1 = Point3D([1, 10, 4])
    p1.plot3d(ax)

    axis = [1., 1., 1.]
    theta = np.pi / 6
    R = geo.rodriguez(axis, theta)
    t = [-4, -4, 0]

    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera(R, t)
    camera1.show(ax)
    camera2.show(ax)

    p2d, p3d = camera2.project([p1])
    camera1.show_projection(ax, [p1])
    camera2.show_projection(ax, [p1])
    print(p2d)

    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    ax.set_zlim([-3, 10])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
