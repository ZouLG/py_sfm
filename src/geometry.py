import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point import *


def quadratic_form(n1, A, n2):
    # n1.T * A * n2
    return np.matmul(np.matmul(n1.reshape((1, -1)), A), n2.reshape(-1, 1))


def homo_rotation_mat(R, t):
    """
    get the homogeneous transformation matrix
        R: 3x3 rotation matrix
        t: 3x1 shift vector
        return: a 4x4 transformation matrix
    """
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.0
    return T


def rodriguez(axis, theta):
    """
    get the rotation matrix via rodriguez formula
        axis: rotation axis
        theta: rotated angle
        return: a rotation matrix
    """
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


def point2line_distance(p, p0, n0):
    delt = p - p0
    n0 = np.array(n0)
    n0_norm = n0 / np.linalg.norm(n0)
    N = cross_mat(n0_norm)
    v = np.matmul(N, delt)
    return np.linalg.norm(v)


def line2line_distance(n1, p1, n2, p2):
    """
    calculate the distance between 3d lines
        n1: the direction vector of the first line
        p1: a point lies in the first line
        n2: the direction vector of the second line
        p2: a point lies in the second line
        return: the distance
    """
    n1 = n1 / np.linalg.norm(n1)
    # n2 = n2 / np.linalg.norm(n2)
    N = np.eye(3) - np.matmul(n1.reshape((3, 1)), n1.reshape((1, 3)))
    delt = p2 - p1
    a = quadratic_form(n2, N, n2)
    b = quadratic_form(delt, N, n2)
    c = quadratic_form(delt, N, delt)
    if a == 0:    # two lines are parallel
        a = 1e-3
    d2 = np.abs(c - b * b / a)
    return np.sqrt(d2)[0]


def pca(mat):
    m, n = mat.shape
    center = np.zeros((m,))
    for i in range(n):
        center += mat[:, i]
    center = center / n

    X = mat
    for i in range(n):
        X[:, i] -= center
    cov_mat = np.matmul(X, X.T) / n
    lamda, eig_V = np.linalg.eig(cov_mat)
    # print(np.matmul(cov_mat, eig_V[:, 0]))
    # print(lamda[0] * eig_V[:, 0])
    return lamda, eig_V


def plot_quadratic_form(A, d, xlim=10, ylim=10, width=200, height=200):
    img = np.zeros((height, width))
    px = np.linspace(-xlim, xlim, width)
    py = np.linspace(-ylim, ylim, height)
    for i in range(height):
        for j in range(width):
            p = np.array((px[j], py[i])).reshape((2, 1))
            qd = quadratic_form(p, A, p)
            if abs(qd - d) < 1:
                img[i, j] = 255
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    A = np.array([[2.0, -0.9], [-0.9, 1.1]])
    plot_quadratic_form(A, 50, xlim=10, ylim=10, width=200, height=200)

    A = [[-1, 1, 0],
         [-4, 3, 0],
         [1, 0, 2]]
    lamda, eig_vec = np.linalg.eig(A)
    print(lamda)
    print(eig_vec)
    print(lamda[1] * eig_vec[:, 1])
    print(np.matmul(A, eig_vec[:, 1]))