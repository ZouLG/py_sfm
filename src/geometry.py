import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point import *


# return a 4x4 homogeneous rigid transformation matrix
def homo_rotation_mat(R, t):
    R_ = np.zeros((4, 4))
    R_[0:3, 0:3] = R
    R_[0:3, 3] = t
    R_[3, 3] = 1.0
    return R_


# get the rotation matrix with rodriguez formula
def rodriguez(axis, theta):
    k = np.array(axis)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0., -k[0]],
                  [-k[1], k[0], 0.]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)
    U, _, V = np.linalg.svd(R)
    return np.matmul(U, V)


def rotate3d(p, *args):
    p_ = Point3D(p)
    if len(args) == 1:
        T = args[0]
        q = np.matmul(T, p_.ph)
    elif len(args) == 2:
        R = args[0]
        t = args[1]
        q = np.matmul(R, p_.p) + t

    if isinstance(p, Point3D):
        return Point3D(q)
    else:
        return q


def cross_mat(n):
    """
        n: a vector of shape (3,)
        return: cross matrix of vector n
    """
    n_ = n.reshape((3,))
    return np.array([[0.0, -n_[2], n_[1]],
                    [n_[2], 0.0, -n_[0]],
                    [-n_[1], n_[0], 0.0]])


def ransac_f_mat(x1, x2, expect=0.0, eps=1e-5, max_iter=100):
    """
        x1: Nx3 homogeneous coordinate of camera1
        x2: Nx3 homogeneous coordinate of camera2
        expect: x2.T * F * x1 should equal to expected value, here is 0.0
        eps: the threshold for inliers and outliers
        max_iter: num of iteration
        return: fundanmental matrix
    """
    def triangulation(F, x1, x2):
        return np.matmul(np.matmul(x2.reshape(1, -1), F), x1.reshape((-1, 1)))

    def solve_f(x1, x2, index):
        N = x1.shape[0]
        A = np.zeros((N, 9))
        for i in index:
            A[i, :] = np.matmul(x2[i, :].reshape((3, 1)), x1[i, :].reshape((1, 3))).reshape((9,))
        _, _, V = np.linalg.svd(A)
        F = V[8, :].reshape((3, 3))     # the eigenvector of the smallest eigen-value
        U, S, V = np.linalg.svd(F)
        S[2] = 0.0                      # set the smallest eigen-value 0 to make rank(F) = 8
        D = np.diag(S)
        F = np.matmul(np.matmul(U, D), V)
        return F

    def get_inliers(F, x1, x2, expect, eps):
        inliers = []
        for i in range(x1.shape[0]):
            diff = np.square(triangulation(F, x1[i, :], x2[i, :]) - expect)
            if diff < eps:
                inliers.append(i)
        return inliers

    num = 8
    N = x1.shape[0]
    inlier_best = []
    F_best = np.eye(3)
    for i in range(max_iter):
        index = random.sample(range(N), num)
        F = solve_f(x1, x2, index)
        inliers = get_inliers(F, x1, x2, expect, eps)
        if len(inliers) > len(inlier_best):
            inlier_best = inliers
            F_best = F

    if len(inlier_best) <= 0:
        print("Error: ransac failed")
        exit()

    # iteration: use inliers to finetune F matrix
    inliers = inlier_best
    F = F_best
    pre_len = 0
    while len(inliers) > pre_len:
        pre_len = len(inliers)
        inlier_best = inliers
        F_best = F
        F = solve_f(x1, x2, inlier_best)
        inliers = get_inliers(F, x1, x2, expect, eps)
        print("inliers number: %d" % (len(inlier_best)))
    return F_best


def decompose_essential_mat(E, *args):
    U, Z, V = np.linalg.svd(E)
    t = U[:, 2]                 # t belongs to the null space of E.T
    Rz = np.array([[0., -1., 0.],
                   [1., 0., 0.],
                   [0., 0., 1.]])
    R1 = np.matmul(np.matmul(U, Rz), V)
    R2 = np.matmul(np.matmul(U, Rz.T), V)
    return [R1, R2],  [t, -t]


if __name__ == "__main__":
    import camera as ca
    f = 1.0
    sx = 0.002
    sy = 0.002
    sx = sx / f
    sy = sy / f
    W = 1920
    H = 1080
    theta = np.pi * 0.9
    axis = np.array([-0.5, 1., 1.])
    R = rodriguez(axis, theta)
    t = np.array([0, -3, 6])
    E = np.matmul(cross_mat(t), R)
    K_ = np.array([[-sx, 0.0, H / 2 * sx],
                   [0.0, sy, -W / 2 * sy],
                   [0.0, 0.0, 1.0]])
    K = np.linalg.pinv(K_)
    F_ = np.matmul(np.matmul(K_.T, E), K_)

    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    p2d1 = np.column_stack((p2d1, np.ones((p2d1.shape[0], 1))))
    p2d2 = np.column_stack((p2d2, np.ones((p2d2.shape[0], 1))))

    p3d1 = ca.back_project(K_, p2d1)
    p3d2 = ca.back_project(K_, p2d2)

    F = ransac_f_mat(p2d1, p2d2, eps=1e-3)
    for i in range(p2d1.shape[0]):
        loss = np.matmul(np.matmul(p2d2[i, :].reshape(1, -1), F), p2d1[i, :].reshape(-1, 1))
        print(loss)

    E = np.matmul(np.matmul(K.T, F), K)
    R_list, t_list = decompose_essential_mat(E)
    R_, t_ = ca.check_validation_rt(R_list, t_list, p3d1, p3d2)
    print(R)
    print(R_)
    print(t_)
