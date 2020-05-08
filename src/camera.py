import numpy as np
import random
from point import *
import geometry as geo
import cv2


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


def camera_triangulation(camera1, camera2, p2d1, p2d2):
    """
    get the back-projected 3D points from the 2D key-points in two different views
        camera1 & camera2: the two cameras
        p2d1: Nx2 array which stores the 2D key-points of camera1
        p2d2: Nx2 array which stores the 2D key-points of camera2
        return: p3d1 & p3d2 is the list of 3D points lie in the two back-projected rays, and p3d lies in the middle
    """
    if p2d1.shape != p2d2.shape:
        print("Error: image key points mismatch")
        exit()
    p3dc1 = camera1.project_image2camera(p2d1)
    p3dc2 = camera2.project_image2camera(p2d2)
    p3dw1 = camera1.project_camera2world(p3dc1)
    p3dw2 = camera2.project_camera2world(p3dc2)
    c1 = camera1.get_camera_center()
    c2 = camera2.get_camera_center()
    p3d1 = []
    p3d2 = []
    p3d = []
    for p1, p2 in zip(p3dw1, p3dw2):
        n1 = p1 - c1
        n2 = p2 - c2
        P, P1, P2 = triangulation(n1, n2, c1, c2)
        p3d.append(P)
        p3d1.append(P1)
        p3d2.append(P2)
    return p3d, p3d1, p3d2


def update_camera_plane(camera):
    img_center = camera.img_center
    img_w = camera.sy * camera.img_w
    img_h = camera.sx * camera.img_h
    p0 = img_center - camera.ex * img_w / 2 - camera.ey * img_h / 2
    p1 = img_center + camera.ex * img_w / 2 - camera.ey * img_h / 2
    p2 = img_center + camera.ex * img_w / 2 + camera.ey * img_h / 2
    p3 = img_center - camera.ex * img_w / 2 + camera.ey * img_h / 2
    return ImgPlane(p0, p1, p2, p3)


class PinHoleCamera:
    def __init__(self, R=np.eye(3),
                 t=np.zeros((3,)),
                 f=1.0, sx=0.002, sy=0.002,
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
        self.__dict__['K'] = np.array([[f / sx, 0.0, img_w / 2],
                                       [0.0, f / sy, img_h / 2],
                                       [0.0, 0.0, 1.0]])
        self.__dict__['K_'] = np.linalg.pinv(self.K)

    def __setattr__(self, key, value):
        if key in ['f', 'sx', 'sy', 'img_w', 'img_h']:
            self.__dict__[key] = value
            self.__dict__['img_center'] = self.o.p + self.f * self.ez
            self.__dict__['img'] = update_camera_plane(self)
            self.__dict__['K'] = np.array([[self.f / self.sx, 0.0, self.img_w / 2],
                                           [0.0, self.f / self.sy, self.img_h / 2],
                                           [0.0, 0.0, 1.0]])
            self.__dict__['K_'] = np.linalg.pinv(self.K)
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
        elif key == "K":
            self.__dict__['K'] = value
            self.__dict__['K_'] = np.linalg.pinv(self.K)
            self.__dict__['f'] = value[0][0] * self.sx
            self.__dict__['img_w'] = value[0][2] * 2
            self.__dict__['img_h'] = value[1][2] * 2
        else:
            print("Caution: Attribute %s can not be set" % (key))

    @staticmethod
    def place_a_camera(p, ez, ex, f=1.0, sx=0.002, sy=0.002, img_w=1920, img_h=1080):
        ez /= np.linalg.norm(ez)
        ex = ex - np.matmul(ez, ex) * ez
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        R = np.column_stack((ex, ey, ez)).T
        t = -np.matmul(R, Point3D(p).p)
        return PinHoleCamera(R, t, f=f, sx=sx, sy=sy, img_w=img_w, img_h=img_h)

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

    def project_image2camera(self, p2d):
        """
        back-project the 2d image point to the homogeneous coordinate in camera frame
            p2d: Nx2 array of 2d key points
            return: list of homogeneous 3d points of in the camera frame
        """
        p2d = np.column_stack((p2d, np.ones((p2d.shape[0], 1))))
        p3d_tmp = np.matmul(self.K_, p2d.T)
        p3d = []
        for i in range(p3d_tmp.shape[1]):
            p3d.append(Point3D(p3d_tmp[:, i]))
        return p3d

    def project_camera2image(self, p3d):
        """
        project the 3d points in the camera frame to 2d image frame
            p3d: list of 3d Points in the camera frame
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
        back-project 3d points in the camera-frame to the world frame
            plist: list of 3d points of type Point3D
            return: list of 3d points of in the world frame
        """
        pw = []
        for p in pc:
            q = p - self.t
            pw.append(Point3D(np.matmul(self.R.T, q.p)))
        return pw

    def project_world2camera(self, pw):
        """
        project 3d points in the world-frame to the camera frame
            plist: list of 3d points of type Point3D in the world-frame
            return: list of 3d points of in the camera frame
        """
        pc = []
        for p in pw:
            pc.append(geo.rotate3d(p, self.T))
        return pc

    def project_world2image(self, pw):
        """
        project 3d points in the world-frame to the 2d image-frame
            plist: list of 3d points of type Point3D in the world-frame
            return: Nx2 array in which each line is a projecting coordinate
        """
        pc = self.project_world2camera(pw)
        pi = self.project_camera2image(pc)
        return pi, pc

    def is_out_of_bound(self, pi):
        return pi[0] < 0 or pi[0] >= self.img_w or pi[1] < 0 or pi[1] >= self.img_h

    def show_projection(self, ax, pw):
        pi, pc = self.project_world2image(pw)
        for i in range(len(pw)):
            o = geo.rotate3d(self.o, self.T_)       # back project to the world frame
            X = pw[i]
            x = pc[i] / pc[i].z
            xi = pi[i]
            X.plot3d(ax, s=10, marker='o', color='blue')
            if not self.is_out_of_bound(xi):
                ax.plot3D([o.x, X.x], [o.y, X.y], [o.z, X.z], color='blue', linestyle='--', linewidth=1)
                xw = geo.rotate3d(x, self.T_)
                xw.plot3d(ax, s=10, marker='o', color='red')
            else:
                ax.plot3D([o.x, X.x], [o.y, X.y], [o.z, X.z], color='red', linestyle='--', linewidth=1)

    def show_projected_img(self, plist):
        """
        draw the projected key-points with circles on a image
        """
        img = np.zeros((self.img_h, self.img_w))
        pi, _ = self.project_world2image(plist)
        for i in range(pi.shape[0]):
            x = int(pi[i][0])
            y = int(pi[i][1])
            cv2.circle(img, (x, y), 15, (255, 255, 255), 4)
        return img

    def get_camera_center(self):
        """
        get the coordinate of the camera center in the world-frame
        """
        return Point3D(-np.matmul(self.R_, self.t))

    def get_essential_mat(self):
        return np.matmul(geo.cross_mat(self.t), self.R)

    def calc_projection_error(self, pw, pi):
        """
        project the 3d points in world-frame, calculate error between the projected 2d coordinates
            with the real image coordinates
        """
        pi_, _ = self.project_world2image(pw)
        err = np.linalg.norm(pi - pi_) / len(pw)
        return err

    def estimate_pose_p4p(self, pw, pi):
        def recover_Rt(pc, pw):
            n = len(pc)
            center_c = Point3D((0, 0, 0))
            center_w = center_c
            for p, q in zip(pc, pw):
                center_c += p
                center_w += q
            center_c /= n
            center_w /= n
            mc = list2mat(pc)
            mw = list2mat(pw)
            mc_c = mc - np.tile(center_c.p, (n, 1))
            mw_c = mw - np.tile(center_w.p, (n, 1))
            R = np.matmul(np.linalg.pinv(np.matmul(mw_c.T, mw_c)), np.matmul(mw_c.T, mc_c))
            U, Z, V = np.linalg.svd(R)
            R = np.matmul(U, V)
            ta = mc - np.matmul(mw, R)
            t = np.mean(ta, axis=0)
            return R.T, t

        # S = [ s0s0, s0s1, s0s2, s0s3, s1s1, s1s2, s1s3, s2s2, s2s3, s3s3 ]
        pc = self.project_image2camera(pi)
        dot_c = np.zeros((4, 4))
        dot_w = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                dot_c[i, j] = pc[i] * pc[j]   # 计算一半即可
                dot_w[i, j] = pw[i] * pw[j]

        L = np.zeros((6, 10))
        b = np.zeros((6, 1))
        idx = 0
        i_range = [0, 0, 0, 1, 1, 2]
        j_range = [1, 2, 3, 2, 3, 3]
        k_range = [1, 2, 3, 2, 3, 3]
        table4 = geo.get_idx_table(4)
        for i, j, k in zip(i_range, j_range, k_range):
            L[idx, table4[i, i]] += dot_c[i, i]
            L[idx, table4[i, j]] += -dot_c[i, j]
            L[idx, table4[i, k]] += -dot_c[i, k]
            L[idx, table4[j, k]] += dot_c[j, k]
            b[idx] = dot_w[i, i] + dot_w[j, k] - dot_w[i, j] - dot_w[i, k]
            idx += 1
        L = np.column_stack([L, -b])
        U, Z, V = np.linalg.svd(L)
        M = V[6:, :].T
        a1 = geo.solve_re_linearization(M[:-1, :], 4)
        S2 = np.matmul(M, a1.reshape(-1, 1))
        S2 = S2 / S2[-1]    # scale to make the last element equal to 1
        S1 = geo.get_first_order(S2[0:10], table4)
        for i in range(len(pc)):
            pc[i].p *= S1[i]
        R, t = recover_Rt(pc, pw)
        return R, t

    def rotate_around_axis(self, theta, axis=0):
        if axis == 0:
            n = self.ex
        elif axis == 1:
            n = self.ey
        else:
            n = self.ez
        Ra_ = geo.rodriguez(n, theta).T
        t = np.matmul(Ra_, self.t)
        R = np.matmul(Ra_, self.R)
        self.R = R
        self.t = t

    def shift_by_t(self, dt):
        self.t = self.t - np.matmul(self.R, np.array(dt))

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


def get_null_space_ransac(x1, x2, eps=1e-5, max_iter=100):
    """
        x1: Nx3 matrix of camera1 points in camera-frame
        x2: Nx3 matrix of camera2 points in camera-frame
        eps: the threshold that distinct inliers and outliers
        max_iter: num of iteration
        return: the null space matrix E
    """
    def calc_loss(E, x1, x2):
        return np.matmul(np.matmul(x2.reshape(1, -1), E), x1.reshape((-1, 1)))

    def solve_ls_fitting(x1, x2, index):
        N = x1.shape[0]
        A = np.zeros((N, 9))
        for i in index:
            A[i, :] = np.matmul(x2[i, :].reshape((3, 1)), x1[i, :].reshape((1, 3))).reshape((9,))
        _, _, V = np.linalg.svd(A)
        F = V[8, :].reshape((3, 3))     # the eigenvector of the smallest eigen-value
        return F

    def project_to_essential_space(E):
        U, S, V = np.linalg.svd(E)
        sigma = (S[0] + S[1]) / 2
        D = np.diag([sigma, sigma, 0.0])
        e = np.matmul(np.matmul(U, D), V)
        return e

    def get_inliers(E, x1, x2, eps):
        inliers = []
        for i in range(x1.shape[0]):
            diff = np.square(calc_loss(E, x1[i, :], x2[i, :]))
            if diff < eps:
                inliers.append(i)
        return inliers

    batchNum = 8
    N = x1.shape[0]
    inlier_best = []
    E_best = np.eye(3)
    for i in range(max_iter):
        index = random.sample(range(N), batchNum)
        E = solve_ls_fitting(x1, x2, index)
        E = project_to_essential_space(E)
        inliers = get_inliers(E, x1, x2, eps)
        if len(inliers) > len(inlier_best):
            inlier_best = inliers
            E_best = E

    assert len(inlier_best) > 0

    # iteration: use inliers to refine E matrix
    inliers = inlier_best
    E = E_best
    pre_len = 0
    while len(inliers) > pre_len:
        pre_len = len(inliers)
        inlier_best = inliers
        E_best = E
        E = solve_ls_fitting(x1, x2, inlier_best)
        E = project_to_essential_space(E)
        inliers = get_inliers(E, x1, x2, eps)
        print("inliers number: %d" % (len(inlier_best)))
    return E_best, inlier_best


def decompose_essential_mat(E):
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
            camera2 = PinHoleCamera(R, t)
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
    print(inliers)
    return Rlist[jopt], tlist[iopt]


#######################################################
#                      test cases
#######################################################
def test_camera_func():
    plt.figure()
    ax = plt.gca(projection='3d')
    p1 = Point3D([5, 5, 7])
    p1.plot3d(ax, s=5)

    camera1 = PinHoleCamera()
    camera1.show(ax)
    camera1.show_projection(ax, [p1])

    plt.figure()
    img = camera1.show_projected_img([p1])
    plt.imshow(img, cmap='gray')

    p3d1 = camera1.project_world2camera([p1])
    p2d1 = camera1.project_camera2image(p3d1)
    print(p3d1)
    print(p2d1)

    ax.set_xlim([-3, 10])
    ax.set_ylim([-3, 10])
    ax.set_zlim([-3, 10])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def test_decompose():
    camera = PinHoleCamera()
    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    print(p2d1)
    print(p2d2)
    p3d1 = camera.project_image2camera(p2d1)
    p3d2 = camera.project_image2camera(p2d2)
    E, _ = get_null_space_ransac(list2mat(p3d1), list2mat(p3d2), eps=1e-3, max_iter=40)

    R_list, t_list = decompose_essential_mat(E)
    R_, t_ = check_validation_rt(R_list, t_list, p3d1, p3d2)
    t_ = t_ * 15
    print("E = \n", E)
    print("R_list = \n", R_list)
    print("t_list = \n", t_list)
    print("R_ = \n", R_)
    print("t_ = \n", t_)
    R_.tofile("../Data/R.dat")
    t_.tofile("../Data/t.dat")


def test_impact_f():
    def show_back_project(ax, camera, p2d, marker='o', color='blue'):
        camera.show(ax, color=color)
        p3dc = camera.project_image2camera(p2d)
        p3dw = camera.project_camera2world(p3dc)
        for p in p3dw:
            p = p * camera.f
            p.plot3d(ax, marker=marker, color=color, s=10)

    plt.figure()
    ax = plt.gca(projection='3d')
    p2d = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))

    camera1 = PinHoleCamera(f=1.0)
    camera2 = PinHoleCamera(f=2.0, t=np.array((0, 0, 1)))
    show_back_project(ax, camera1, p2d, marker='o', color='red')
    show_back_project(ax, camera2, p2d, marker='x', color='blue')

    lim_low = -1
    lim_high = 3
    ax.set_xlim([lim_low, lim_high])
    ax.set_ylim([lim_low, lim_high])
    ax.set_zlim([lim_low, lim_high])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    test_decompose()
    # test_camera_func()
    # test_impact_f()
