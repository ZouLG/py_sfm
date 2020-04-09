from point import *
import geometry as geo
import numpy as np


def get_control_points(pw):
    p0 = geo.calc_center(pw)
    lamda, eig_v = geo.pca(list2mat(pw).T)
    p1 = p0 + np.sqrt(lamda[0]) * eig_v[0]
    p2 = p0 + np.sqrt(lamda[1]) * eig_v[1]
    p3 = p0 + np.sqrt(lamda[2]) * eig_v[2]
    # p1 = Point3D((1, 0, 0))
    # p2 = Point3D((0, 1, 0))
    # p3 = Point3D((0, 0, 1))
    return p0, p1, p2, p3


def calc_barycentric(ctrl_pts, pw):
    m = len(ctrl_pts)
    n = len(pw)
    base = list2mat(ctrl_pts)
    base = np.column_stack((base, np.ones((m,))))  # sum(bc) = 1
    base_inv = np.linalg.pinv(np.matmul(base, base.T))
    mat_pw = np.column_stack((list2mat(pw), np.ones(n, )))

    bary_coef = np.matmul(np.matmul(mat_pw, base.T), base_inv)
    # print(list2mat(ctrl_pts).T)
    # print(bary_coef.T)
    # print(np.matmul(list2mat(ctrl_pts).T, bary_coef.T).T)
    return bary_coef


def calc_dist(pw):
    n = len(pw)
    idx = 0
    dist = np.zeros(geo.sum_to_n(n - 1), )
    for i in range(n):
        for j in range(i + 1, n):
            dist[idx] = np.linalg.norm(pw[i] - pw[j])
            idx += 1
    return dist


def calc_sign(ctrl_pc):
    depth = np.array([p.z for p in ctrl_pc])
    print(depth)
    if np.sum(depth > 0.0) > len(ctrl_pc) // 2:
        return 1
    else:
        return -1


def estimate_pose_n1(v, ctrl_pw):
    ctrl_pc = []
    for i in range(0, len(ctrl_pw)):
        ctrl_pc.append(Point3D(v[i * 3: i * 3 + 3]))
    dist_c = calc_dist(ctrl_pc)
    dist_w = calc_dist(ctrl_pw)
    scale = calc_sign(ctrl_pc) * np.matmul(dist_w, dist_c) / np.matmul(dist_c, dist_c)
    for i in range(len(ctrl_pc)):
        ctrl_pc[i].p = ctrl_pc[i].p * scale
    R, t = geo.recover_Rt(ctrl_pc, ctrl_pw)
    return R, t


def estimate_pose_n234(v, ctrl_pw, N=2):
    n = len(ctrl_pw)
    V = np.split(v, n)
    m = geo.sum_to_n(n - 1)
    L = np.zeros((m, geo.sum_to_n(N)))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dv = V[i] - V[j]
            L[idx, :] = geo.trans_mat2vec(np.matmul(dv.T, dv))
            idx += 1
    dist_w = calc_dist(ctrl_pw)
    dist_w = dist_w * dist_w
    if L.shape[0] < L.shape[1]:
        uu, z, vv = np.linalg.svd(np.column_stack((L, -dist_w)))
        vv = vv[-(L.shape[1] - L.shape[0] + 1):, :].T
        a = geo.solve_re_linearization(vv, N)
        beta2 = np.matmul(vv, a)
        beta2 /= beta2[-1]
        beta2 = beta2[:-1]
    else:
        L_pinv = np.linalg.pinv(np.matmul(L.T, L))
        beta2 = np.matmul(L_pinv, np.matmul(L.T, dist_w))
    beta1 = geo.get_first_order(beta2, geo.get_idx_table(N))
    ctrl_pc = [Point3D(np.matmul(pc, beta1)) for pc in V]
    sign = calc_sign(ctrl_pc)
    ctrl_pc = [pc * sign for pc in ctrl_pc]
    R, t = geo.recover_Rt(ctrl_pc, ctrl_pw)
    return R, t


def estimate_pose_epnp(K, pw, pi, ctrl_num=4):
    n = len(pw)
    ctrl_pw = get_control_points(pw)
    bc = calc_barycentric(ctrl_pw, pw)
    L = np.zeros((n * 2, ctrl_num * 3))
    fu = K[0, 0]
    fv = K[1, 1]
    uc = K[0, 2]
    vc = K[1, 2]
    for i in range(n):
        for j in range(ctrl_num):
            L[i * 2, j * 3] = fu * bc[i, j]
            L[i * 2, j * 3 + 2] = (uc - pi[i, 0]) * bc[i, j]
            L[i * 2 + 1, j * 3 + 1] = fv * bc[i, j]
            L[i * 2 + 1, j * 3 + 2] = (vc - pi[i, 1]) * bc[i, j]
    u, z, eig_v = np.linalg.svd(L)
    eig_v = eig_v.T
    # print(z)
    # M = np.matmul(L.T, L)
    # lamda, eig_v = np.linalg.eig(M)
    # print(lamda)
    R, t = estimate_pose_n1(eig_v[:, -1], ctrl_pw)
    # R, t = estimate_pose_n234(eig_v[:, -2:], ctrl_pw)
    # R, t = estimate_pose_n234(eig_v[:, -3:], ctrl_pw, 3)
    # R, t = estimate_pose_n234(eig_v[:, -4:], ctrl_pw, 4)
    print(R)
    print(t)
    return R, t
