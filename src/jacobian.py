import numpy as np
from quarternion import Quarternion


def derr_over_dpc(q, t, fu, fv, pw):
    pc = Quarternion.rotate_with_quaternion(q, pw.p) + t
    return np.array([[fu, 0, -fu * pc[0] / pc[2]],
                     [0, fv, -fv * pc[1] / pc[2]]]) / pc[2]


def derr_over_dcam(q, t, fu, fv, pw):
    """
        2x7 jacobian matrix of derivative err to camera pose, 4 for q, 3 for t
    """
    n = q.norm()
    dedpc = derr_over_dpc(q, t, fu, fv, pw)
    dpcdr = np.zeros((3, 9))
    dpcdr[0, 0:3] = pw.p
    dpcdr[1, 3:6] = pw.p
    dpcdr[2, 6:9] = pw.p
    q0, q1, q2, q3 = q / n
    drdq = 2 * np.array([[0, 0, -2 * q2, -2 * q3],
                         [-q3, q2, q1, -q0],
                         [q2, q3, q0, q1],
                         [q3, q2, q1, q0],
                         [0, -2 * q1, 0, -2 * q3],
                         [-q1, -q0, q3, q2],
                         [-q2, q3, -q0, q1],
                         [q1, q0, q3, q2],
                         [0, -2 * q1, -2 * q2, 0]])
    qq = q.q
    dqdu = n ** 2 * np.eye(4) - np.matmul(qq.reshape((-1, 1)), qq.reshape(1, -1))
    dqdu /= (n ** 3)
    jq = np.matmul(np.matmul(np.matmul(dedpc, dpcdr), drdq), dqdu)
    jcam = np.concatenate((jq, dedpc), axis=1)
    return jcam


def derr_over_df(q, t, pw):
    pc = Quarternion.rotate_with_quaternion(q, pw.p) + t
    return np.diag([pc[0] / pc[2], pc[1] / pc[2]])


def derr_over_dpw(q, t, fu, fv, pw):
    dedpc = derr_over_dpc(q, t, fu, fv, pw)
    dpcdpw = Quarternion.quaternion_to_mat(q)
    return np.matmul(dedpc, dpcdpw)
