import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from point import *


def calc_min_dis(a, b):
    min_dis = np.Inf
    for p in a:
        for q in b:
            dis = np.linalg.norm(p, q)
            if dis < min_dis:
                min_dis = dis
    return min_dis


def sum_to_n(n):
    return (1 + n) * n // 2


def quadratic_form(n1, A, n2):
    # n1.T * A * n2
    return np.squeeze(np.matmul(np.matmul(n1.reshape((1, -1)), A), n2.reshape(-1, 1)))


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
    k = np.squeeze(np.asarray(axis))
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0., -k[0]],
                  [-k[1], k[0], 0.]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)
    U, _, V = np.linalg.svd(R)
    return np.matmul(U, V)


def rigid_transform(p, R, t):
    p_ = Point3D(p)
    q = np.matmul(R, p_.p) + t
    if isinstance(p, Point3D):
        return Point3D(q)
    else:
        return q


def rigid_inv_transform(p, R, t):
    p_ = Point3D(p)
    q = np.matmul(R.T, p_.p - t)
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


def calc_center(plist):
    center = Point3D((0, 0, 0))
    for p in plist:
        center += p
    center = center / len(plist)
    return center


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


def get_idx_table(n):
    table = np.zeros((n, n), np.int)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            table[i, j] = idx
            table[j, i] = idx
            idx += 1
    return table


def get_first_order(s2, table):
    def check_s1(s1, s2):
        n = len(s1)
        s2_ = np.zeros(s2.shape)
        for i in range(n):
            for j in range(i, n):
                s2_[table[i, j]] = s1[i] * s1[j]
        err = np.max(np.abs(s2_ - s2))
        if err > 0.0001:
            return False
        return True

    m = len(s2)
    n = int((np.sqrt(m * 8 + 1) - 1) / 2)
    s2 = s2 * np.sign(s2[0])
    s1 = np.zeros((n,))
    s1[0] = np.sqrt(s2[0])
    for i in range(1, n):
        s1[i] = s2[i] / s1[0]
    if check_s1(s1, s2):
        print("Warning: no solution")
    return s1


def trans_mat2vec(mat):
    dim = mat.shape[0]
    vec_len = sum_to_n(dim)
    vec = np.zeros((vec_len,))
    idx = 0
    for i in range(dim):
        for j in range(i, dim):
            vec[idx] = mat[i, j] + mat[j, i] * (j > i)
            idx += 1
    return vec


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


def solve_re_linearization(M, s_dim):
    def get_permutation_list(n):
        table = get_idx_table(n)
        permu_list = []
        for i in range(0, n - 1):
            for j in range(i, n - 1):
                for k in range(i + 1, n):
                    for l in range(k, n):
                        if (j == k and k == l) or j > k:
                            continue
                        if j == k:
                            # print(i, j, k, l)
                            permu_list.append((table[i, j], table[k, l], table[i, l], table[j, k]))
                        else:
                            # print(i, j, k, l)
                            permu_list.append((table[i, j], table[k, l], table[i, k], table[j, l]))
                        if (i < j) and (j < k) and (k < l):
                            # print(i, j, k, l)
                            permu_list.append((table[i, j], table[k, l], table[i, l], table[j, k]))
        return permu_list

    def get_coefs(M, i, j, k, l):
        vi = M[i, :].reshape((-1, 1))
        vj = M[j, :].reshape((1, -1))
        vk = M[k, :].reshape((-1, 1))
        vl = M[l, :].reshape((1, -1))
        # print(vi.shape, vj.shape, vk.shape, vl.shape)
        return np.matmul(vi, vj) - np.matmul(vk, vl)

    m, n = M.shape
    permu_list = get_permutation_list(s_dim)
    L = np.zeros((len(permu_list), sum_to_n(n)))
    for i in range(len(permu_list)):
        p = permu_list[i]
        L[i, :] = trans_mat2vec(get_coefs(M, p[0], p[1], p[2], p[3]))
    u, z, v = np.linalg.svd(L)
    S2 = v[-1, :]
    S1 = get_first_order(S2, get_idx_table(n))
    return S1


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