import cv2
import random
import numpy as np
import geometry as geo
from point import Point3D, list2mat
from quarternion import Quarternion

__all__ = [
    "PinHoleCamera",
    "camera_triangulation",
    "get_null_space_ransac",
    "check_validation_rt",
    "decompose_essential_mat",
    "ransac_estimate_pose",
    "ransac_estimate_pose_old"
]


def point_project2line(o, n, p):
    delt = p - o
    t = np.matmul(delt, n) / np.matmul(n, n)
    return o + t * n


def bundle_projection(PinHoleCamera, pw, pi):
    pc = PinHoleCamera.project_image2camera(pi)
    pcw = PinHoleCamera.project_camera2world(pc)
    c = PinHoleCamera.get_camera_center()
    pw_new = []
    for q, p in zip(pcw, pw):
        n = q - c
        pw_new.append(point_project2line(c, n, p))
    return pw_new


def triangulation(n1, n2, p1, p2):
    delt = p2 - p1
    N1 = np.matmul(n1, n1) * np.eye(3) - np.matmul(n1.reshape(3, 1), n1.reshape((1, 3)))
    N2 = np.matmul(n2, n2) * np.eye(3) - np.matmul(n2.reshape(3, 1), n2.reshape((1, 3)))
    a2 = geo.quadratic_form(n2, N1, delt)
    b2 = geo.quadratic_form(n2, N1, n2)
    t2 = -a2 / b2
    v2 = t2 * n2
    P2 = Point3D(p2 + v2)

    a1 = geo.quadratic_form(n1, N2, -delt)
    b1 = geo.quadratic_form(n1, N2, n1)
    t1 = -a1 / b1
    v1 = t1 * n1
    P1 = Point3D(p1 + v1)
    P = (P1 + P2) / 2
    return P, P1, P2


def camera_triangulation_old(camera1, camera2, p2d1, p2d2):
    """
    get the back-projected 3D points from the 2D key-points in two different views
        camera1 & camera2: the two cameras
        p2d1: Nx2 array which stores the 2D key-points of camera1
        p2d2: Nx2 array which stores the 2D key-points of camera2
        return: p3d1 & p3d2 is the list of 3D points lie in the two back-projected rays, and p3d lies in the middle
    """
    assert p2d1.shape == p2d2.shape
    p3dc1 = camera1.project_image2camera(p2d1)
    p3dc2 = camera2.project_image2camera(p2d2)
    p3dw1 = camera1.project_camera2world(p3dc1)
    p3dw2 = camera2.project_camera2world(p3dc2)
    c1 = camera1.get_camera_center()
    c2 = camera2.get_camera_center()
    p3d, p3d1, p3d2 = [], [], []
    for p1, p2 in zip(p3dw1, p3dw2):
        n1 = p1 - c1
        n2 = p2 - c2
        P, P1, P2 = triangulation(n1, n2, c1, c2)
        p3d.append(P)
        p3d1.append(P1)
        p3d2.append(P2)
    return p3d, p3d1, p3d2


def camera_triangulation(camera1, camera2, pi1, pi2, filter=False):
    """
    get the back-projected 3D points from the 2D key-points in two views
        camera1 & camera2: the two cameras
        pi1 & pi2: Nx2 array which stores the 2D key-points of the two cameras
        filter: ignore points which have small view angles when filter is True
        return: reconstructed points' coordinates in world-frame
    """
    assert pi1.shape == pi2.shape
    pc1 = camera1.project_image2camera(pi1)
    pc2 = camera2.project_image2camera(pi2)
    r = np.matmul(camera2.R, camera1.R.T)
    a = np.matmul(r, camera1.t) - camera2.t
    pw = []
    for p, q in zip(pc1, pc2):
        b = geo.cross_mat(q.p)
        x = np.matmul(np.matmul(b, r), p.p)
        y = np.matmul(b, a)
        s1 = np.matmul(x, y) / np.matmul(x, x)
        pw.append(Point3D(np.matmul(camera1.R.T, s1 * p.p - camera1.t)))
    return pw, None, None


class PinHoleCamera(object):
    def __init__(self, **kwargs):
        # intrinsic params
        f, fx, fy, img_w, img_h = [1.0, 500, 500, 1920, 1080]
        if 'R' in kwargs:
            R = kwargs['R']
            assert R.shape == (3, 3), "rotation mat should be 3x3 matrix"
        else:
            R = np.eye(3)

        if 't' in kwargs:
            t = kwargs['t']
        else:
            t = np.zeros((3,))

        if 'f' in kwargs:
            f = kwargs['f']
        if 'fx' in kwargs:
            fx = kwargs['fx']
        if 'fy' in kwargs:
            fy = kwargs['fy']
        if 'img_w' in kwargs:
            img_w = kwargs['img_w']
        if 'img_h' in kwargs:
            img_h = kwargs['img_h']
        if 'K' in kwargs:
            K = kwargs['K']
            fx = K[0, 0]
            fy = K[1, 1]
            img_w = K[0, 2] * 2
            img_h = K[1, 2] * 2

        self.__dict__['R'] = R
        self.__dict__['t'] = t
        self.__dict__['q'] = Quarternion.mat_to_quaternion(R)
        self.__dict__['f'] = f
        self.__dict__['fx'] = fx
        self.__dict__['fy'] = fy
        self.__dict__['img_w'] = img_w
        self.__dict__['img_h'] = img_h
        self.__dict__['K'] = np.array([[fx, 0.0, img_w / 2],
                                       [0.0, fy, img_h / 2],
                                       [0.0, 0.0, 1.0]])
        self.__dict__['K_'] = np.linalg.pinv(self.K)

    def __setattr__(self, key, value):
        if key == 'fx':
            self.__dict__['K'][0, 0] = value
        elif key == 'fy':
            self.__dict__['K'][1, 1] = value
        elif key == 'img_w':
            self.__dict__['K'][0, 2] = value / 2
        elif key == 'img_h':
            self.__dict__['K'][1, 2] = value / 2
        elif key == 'K':
            self.__dict__['fx'] = value[0, 0]
            self.__dict__['fy'] = value[1, 1]
            self.__dict__['img_w'] = value[0, 2] * 2
            self.__dict__['img_h'] = value[1, 2] * 2
        elif key == 'q':
            self.__dict__['q'] = Quarternion(value)
            self.__dict__['R'] = Quarternion.quaternion_to_mat(value)
        elif key == 'R':
            self.__dict__['R'] = value
            self.__dict__['q'] = Quarternion.mat_to_quaternion(value)
        elif key not in ['f', 't']:
            raise AttributeError
        self.__dict__[key] = value
        self.__dict__['K_'] = np.linalg.pinv(self.K)

    @staticmethod
    def place_a_camera(p, ez, ex, f=1.0, fx=500, fy=500, img_w=1920, img_h=1080):
        ez /= np.linalg.norm(ez)
        ex = ex - np.matmul(ez, ex) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        R = np.row_stack((ex, ey, ez))
        t = -np.matmul(R, Point3D(p).p)
        return PinHoleCamera(R=R, t=t, f=f, fx=fx, fy=fy, img_w=img_w, img_h=img_h)

    def show(self, ax, color='blue', s=20):
        o = geo.rigid_inv_transform(Point3D((0, 0, 0)), self.R, self.t)
        o.plot3d(ax, color=color, s=s)
        ex = self.R[0, :]
        ey = self.R[1, :]
        ez = self.R[2, :]
        ax.quiver(o.x, o.y, o.z, ex[0], ex[1], ex[2], normalize=True, color='red')
        ax.quiver(o.x, o.y, o.z, ey[0], ey[1], ey[2], normalize=True, color='green')
        ax.quiver(o.x, o.y, o.z, ez[0], ez[1], ez[2], normalize=True, color='blue')

        sx, sy = [self.f / self.fx, self.f / self.fy]
        c = o + ez * self.f
        p0 = c - ex * sx * self.img_w / 2 - ey * sy * self.img_h / 2
        p1 = c + ex * sx * self.img_w / 2 - ey * sy * self.img_h / 2
        p2 = c + ex * sx * self.img_w / 2 + ey * sy * self.img_h / 2
        p3 = c - ex * sx * self.img_w / 2 + ey * sy * self.img_h / 2

        p0.plot3d(ax, color='red', marker='.', s=20)
        p1.plot3d(ax, color=color, marker='.', s=20)
        p3.plot3d(ax, color=color, marker='.', s=20)
        p2.plot3d(ax, color=color, marker='.', s=20)
        ax.plot3D([p0.x, p1.x], [p0.y, p1.y], [p0.z, p1.z], color=color)
        ax.plot3D([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color=color)
        ax.plot3D([p2.x, p3.x], [p2.y, p3.y], [p2.z, p3.z], color=color)
        ax.plot3D([p3.x, p0.x], [p3.y, p0.y], [p3.z, p0.z], color=color)
        ax.plot3D([o.x, p0.x], [o.y, p0.y], [o.z, p0.z], color=color)
        ax.plot3D([o.x, p1.x], [o.y, p1.y], [o.z, p1.z], color=color)
        ax.plot3D([o.x, p2.x], [o.y, p2.y], [o.z, p2.z], color=color)
        ax.plot3D([o.x, p3.x], [o.y, p3.y], [o.z, p3.z], color=color)

    def project_image2camera(self, p2d):
        """
        back-project the 2d image Point3D to the homogeneous coordinate in PinHoleCamera Frame
            p2d: Nx2 array of 2d key points
            return: list of homogeneous 3d points of in the PinHoleCamera Frame
        """
        p2d = np.column_stack((p2d, np.ones((p2d.shape[0], 1))))
        p3d_tmp = np.matmul(self.K_, p2d.T)
        p3d = []
        for i in range(p3d_tmp.shape[1]):
            p3d.append(Point3D(p3d_tmp[:, i]))
        return p3d

    def project_camera2image(self, p3d):
        """
        project the 3d points in the PinHoleCamera Frame to 2d image Frame
            p3d: list of 3d Points in the PinHoleCamera Frame
            return: Nx2 array of 2d key points
        """
        N = len(p3d)
        p2d = np.zeros((N, 2))
        for i in range(N):
            p = p3d[i].p
            q = np.matmul(self.K, p) / p[2]
            p2d[i, :] = q[0:2]
        return p2d

    def project_camera2world(self, pc):
        """
        back-project 3d points in the PinHoleCamera-Frame to the world Frame
            plist: list of 3d points of type Point3D
            return: list of 3d points of in the world Frame
        """
        pw = []
        for p in pc:
            q = geo.rigid_inv_transform(p, self.R, self.t)
            pw.append(q)
        return pw

    def project_world2camera(self, pw):
        """
        project 3d points in the world-Frame to the PinHoleCamera Frame
            plist: list of 3d points of type Point3D in the world-Frame
            return: list of 3d points of in the PinHoleCamera Frame
        """
        pc = []
        for p in pw:
            pc.append(geo.rigid_transform(p, self.R, self.t))
        return pc

    def project_world2image(self, pw):
        """
        project 3d points in the world-Frame to the 2d image-Frame
            plist: list of 3d points of type Point3D in the world-Frame
            return: Nx2 array in which each line is a projecting coordinate
        """
        pc = self.project_world2camera(pw)
        pi = self.project_camera2image(pc)
        return pi, pc

    def is_out_of_bound(self, pi):
        return pi[0] < 0 or pi[0] >= self.img_w or pi[1] < 0 or pi[1] >= self.img_h

    def show_projection(self, ax, pw):
        pi, pc = self.project_world2image(pw)
        for i, p in enumerate(pw):
            o = geo.rigid_inv_transform(Point3D((0, 0, 0)), self.R, self.t)
            x = pc[i] / pc[i].z
            xi = pi[i]
            p.plot3d(ax, s=10, marker='o', color='blue')
            if not self.is_out_of_bound(xi):
                ax.plot3D([o.x, p.x], [o.y, p.y], [o.z, p.z], color='blue', linestyle='--', linewidth=1)
                xw = geo.rigid_inv_transform(x, self.R, self.t)
                xw.plot3d(ax, s=10, marker='o', color='red')
            else:
                ax.plot3D([o.x, p.x], [o.y, p.y], [o.z, p.z], color='red', linestyle='--', linewidth=1)

    def get_projected_img(self, plist):
        """
            draw the projected key-points with circles on a image
        """
        img = np.zeros((self.img_h, self.img_w))
        pi, _ = self.project_world2image(plist)
        for i in range(pi.shape[0]):
            x, y = pi[i, :]
            cv2.circle(img, (int(x), int(y)), 15, (255, 255, 255), 4)
        return img

    def get_camera_center(self):
        """
            get the coordinate of the PinHoleCamera center in the world-Frame
        """
        return Point3D(-np.matmul(self.R.T, self.t))

    def get_essential_mat(self):
        return np.matmul(geo.cross_mat(self.t), self.R)

    def calc_projection_error(self, pw, pi):
        """
        project the 3d points in world-Frame, calculate error between the projected 2d coordinates
            with the real image coordinates
        """
        pi_, _ = self.project_world2image(pw)
        err = np.linalg.norm(pi - pi_) / len(pw)
        return err


def get_null_space_ransac(x1, x2, eps=1e-5, max_iter=500):
    """
        x1: Nx3 matrix of camera1 points in camera-frame
        x2: Nx3 matrix of camera2 points in camera-frame
        eps: the threshold that distinct inliers and outliers
        max_iter: num of iteration
        return: the null space matrix E
    """
    def calc_loss(E, x1, x2):
        return np.matmul(np.matmul(x2.reshape(1, -1), E), x1.reshape((-1, 1)))

    def get_inliers(E, x1, x2, eps):
        inliers = []
        for i in range(x1.shape[0]):
            diff = np.square(calc_loss(E, x1[i, :], x2[i, :]))
            if diff < eps:
                inliers.append(i)
        return inliers

    assert x1.shape == x2.shape, "shape of x1 & x2 is inconsistent."
    random.seed(0)
    batch_num = 8
    num = x1.shape[0]
    inlier_best = []
    Ebest = np.eye(3)
    ibest = 0
    for i in range(max_iter):
        index = random.sample(range(num), batch_num)
        E = geo.solve_ls_fitting(x1, x2, index)
        if E is None:
            continue
        # E = geo.project_to_essential_space(E)
        inliers = get_inliers(E, x1, x2, eps)
        if len(inliers) > len(inlier_best):
            inlier_best = inliers
            Ebest = E
            ibest = i
        if len(inlier_best) > num * 0.7 or i - ibest > 50:
            break
    assert len(inlier_best) > 0, "there are not enough inlier matches"

    # iteration: use inliers to refine E matrix
    inliers = inlier_best
    E = Ebest
    pre_len = 0
    while len(inliers) > pre_len:
        pre_len = len(inliers)
        inlier_best = inliers
        Ebest = E
        E = geo.solve_ls_fitting(x1, x2, inlier_best)
        if E is None:
            continue
        # E = geo.project_to_essential_space(E)
        inliers = get_inliers(E, x1, x2, eps)
    return Ebest, inlier_best


def decompose_essential_mat(E):
    E = geo.project_to_essential_space(E)
    # result of svd isn't unique, E is degenerate(has 2 same singular value, how to decompose E ????)
    U, Z, V = np.linalg.svd(E)
    Rz = np.array([[0., -1., 0.],
                   [1., 0., 0.],
                   [0., 0., 1.]])
    t = U[:, 2]                 # t belongs to the null space of E.T
    # T = np.matmul(np.matmul(np.matmul(U, Rz), np.diag(Z)), U.T)
    # t = np.array([T[1, 2], -T[0, 2], T[0, 1]])
    R1 = np.matmul(np.matmul(U, Rz), V)
    R2 = np.matmul(np.matmul(U, Rz.T), V)
    R1 = np.sign(np.linalg.det(R1)) * R1
    R2 = np.sign(np.linalg.det(R2)) * R2
    return [R1, R2],  [t, -t]


def check_validation_rt(Rlist, tlist, pc1, pc2):
    """
    eliminate invalid R & t decomposed from essential matrix
        R: list of rotation matrix of camera2, length is 2
        t: list of  shift vectors of camera2
        p1: list of 3D coordinates of camera1
        p2: list of 3D coordinates of camera2
        return: the R & t satisfy all Z > 0 constrain
    """
    inliers = np.zeros((len(tlist), len(Rlist)))
    camera1 = PinHoleCamera()
    c1 = camera1.get_camera_center()
    n1 = [x - c1 for x in pc1]
    th = 0
    iopt = 0
    jopt = 0
    for i in range(len(tlist)):
        t = tlist[i]
        for j in range(len(Rlist)):
            R = Rlist[j]
            camera2 = PinHoleCamera(R=R, t=t)
            c2 = camera2.get_camera_center()
            pw2 = camera2.project_camera2world(pc2)
            n2 = [x - c2 for x in pw2]
            for a, b in zip(n1, n2):
                p, p0, p1 = triangulation(a, b, c1, c2)
                if np.matmul(p - c1, a) > 0 and np.matmul(p - c2, b) > 0:
                    inliers[i][j] += 1
            if inliers[i][j] > th:
                th = inliers[i][j]
                iopt = i
                jopt = j
    return Rlist[jopt], tlist[iopt]


def ransac_estimate_pose_old(
    pi_ref,
    pi_mat,
    cam_ref,
    cam_mat,
    eps=1e-4,
    max_iter=300,
    t_scale=100
):
    assert pi_ref.shape == pi_mat.shape
    pc1 = cam_ref.project_image2camera(pi_ref)
    pc2 = cam_mat.project_image2camera(pi_mat)
    try:
        E, inliers = get_null_space_ransac(
            x1=list2mat(pc1),
            x2=list2mat(pc2),
            eps=eps,
            max_iter=max_iter
        )
    except:
        print("Warning: there are not enough matching points")
        return None, None, []

    R_list, t_list = decompose_essential_mat(E)
    R, t = check_validation_rt(R_list, t_list, pc1, pc2)
    t *= t_scale
    return R, t, inliers


def ransac_estimate_pose(
    pi_ref,
    pi_mat,
    cam_ref,
    cam_mat,
    eps=1e-4,
    max_iter=300,
    t_scale=100,
    solve_pose=True
):
    assert pi_ref.shape == pi_mat.shape
    num = len(pi_ref)
    pih1 = np.column_stack((pi_ref, np.ones(num,)))
    pih2 = np.column_stack((pi_mat, np.ones(num,)))
    try:
        F, inliers = get_null_space_ransac(
            x1=pih1,
            x2=pih2,
            eps=eps,
            max_iter=max_iter
        )
    except:
        print("Warning: there are not enough matching points")
        return None, None, []

    E = np.matmul(np.matmul(cam_mat.K.T, F), cam_ref.K)
    if not solve_pose:
        return E, inliers

    Rs, ts = decompose_essential_mat(E)
    pc1 = cam_ref.project_image2camera(pi_ref)
    pc2 = cam_mat.project_image2camera(pi_mat)
    R, t = check_validation_rt(Rs, ts, pc1, pc2)
    t *= t_scale
    return R, t, inliers


#######################################################
#                      test cases
#######################################################
def test_camera_func():
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.gca(projection='3d')
    p1 = Point3D([2, 6, 7])
    p1.plot3d(ax, s=5)

    R = geo.rodriguez((1, 1, 1), np.pi / 4)
    camera1 = PinHoleCamera(R=R)
    camera2 = PinHoleCamera(R=R, t=np.array((-4, -4, 0)))
    camera1.show(ax)
    camera2.show(ax)

    pi1, _ = camera1.project_world2image([p1])
    pi2, _ = camera2.project_world2image([p1])
    pw, _, _ = camera_triangulation(camera1, camera2, pi1, pi2)
    print(pw)

    # pi = np.array([[1920, 1080]])
    # pc = camera1.project_image2camera(pi)
    # pi_ = camera1.project_camera2image(pc)
    # print(pi_)
    # print(pc)
    #
    # pw = camera1.project_camera2world(pc)
    # pc_ = camera1.project_world2camera(pw)
    # print(pc_)
    # pw[0].plot3d(ax, s=5, marker='x', color='red')

    # camera1.show_projection(ax, [p1])
    # img = camera1.get_projected_img([p1])
    # plt.figure()
    # plt.imshow(img)

    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    ax.set_zlim([-3, 10])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def test_decompose():
    from data import generate_rand_points
    pw = generate_rand_points(10, [0, 0, 10], [5, 5, 5])
    camera1 = PinHoleCamera()
    camera2 = PinHoleCamera.place_a_camera((10, 10, 0), (0, 0, 1), (0, 1, 0))

    pi1, pc1 = camera1.project_world2image(pw)
    pi2, pc2 = camera2.project_world2image(pw)

    R1, t1, _ = ransac_estimate_pose_old(pi1, pi2, camera1, camera2)
    R2, t2, _ = ransac_estimate_pose(pi1, pi2, camera1, camera2)
    print("R* = \n", camera2.R)
    print("t* = \n", camera2.t)
    print("R1 = \n", R1)
    print("t1 = \n", t1)
    print("R2 = \n", R2)
    print("t2 = \n", t2)


if __name__ == "__main__":
    # test_camera_func()
    test_decompose()
