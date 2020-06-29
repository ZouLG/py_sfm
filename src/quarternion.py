import numpy as np
from scipy.spatial.transform.rotation import Rotation


class Quarternion(object):
    def __init__(self, value):
        if type(value) in [np.ndarray, tuple, list]:
            if len(value) == 3:
                self.q = np.zeros((4,))
                self.q[1:] = value
            elif len(value) == 4:
                self.q = np.asarray(value)
        elif isinstance(value, Quarternion):
            self.q = value.q

    def __getitem__(self, item):
        return self.q[item]

    def __repr__(self):
        return "[%f, %fi, %fj, %fk]" % (self.q[0], self.q[1], self.q[2], self.q[3])

    def __add__(self, obj):
        if isinstance(obj, Quarternion):
            return Quarternion(self.q + obj.q)
        elif type(obj) in [list, tuple, np.ndarray]:
            return Quarternion(self.q + obj)

    def __sub__(self, obj):
        if isinstance(obj, Quarternion):
            return Quarternion(self.q - obj.q)
        elif isinstance(obj, np.ndarray):
            return Quarternion(self.q - obj)

    def __mul__(self, obj):
        if isinstance(obj, Quarternion):
            imag1 = self.q[1:]
            imag2 = obj[1:]
            quat = np.zeros((4,))
            quat[0] = self.q[0] * obj[0] - np.matmul(imag1, imag2)
            quat[1:] = self.q[0] * imag2 + obj[0] * imag1 + np.cross(imag1, imag2)
            return Quarternion(quat)
        elif type(obj) in [int, float]:
            return Quarternion(self.q * obj)

    def __truediv__(self, dsor):
        return Quarternion(self.q / dsor)

    def norm(self):
        return np.linalg.norm(self.q)

    def inv(self):
        norm2 = np.linalg.norm(self.q) ** 2
        return Quarternion(self.q * (1, -1, -1, -1) / norm2)

    @staticmethod
    def quaternion_to_mat(q):
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        # q0, q1, q2, q3 = q / q.norm()
        # rx = 2 * np.array([0.5 - q2 * q2 - q3 * q3, q1 * q2 - q0 * q3, q1 * q3 + q0 * q2])
        # ry = 2 * np.array([q1 * q2 + q0 * q3, 0.5 - q1 ** 2 - q3 ** 2, q2 * q3 - q0 * q1])
        # rz = 2 * np.array([q1 * q3 - q0 * q2, q2 * q3 + q0 * q1, 0.5 - q1 ** 2 - q2 ** 2])
        # return np.row_stack([rx, ry, rz])

    @staticmethod
    def mat_to_quaternion(R):
        q1, q2, q3, q0 = Rotation.from_matrix(R).as_quat()
        # q0 = np.sqrt(np.trace(R) + 1) / 2
        # q1 = (R[2, 1] - R[1, 2]) / q0 / 4
        # q2 = (R[0, 2] - R[2, 0]) / q0 / 4
        # q3 = (R[1, 0] - R[0, 1]) / q0 / 4
        return Quarternion((q0, q1, q2, q3))

    @staticmethod
    def rotate_with_quaternion(q, p):
        pt = q * Quarternion(p) * q.inv()
        return pt[1:]


if __name__ == "__main__":
    # test rotation with quarternion
    a = Quarternion((1, -1, -1, -1))
    Ra = Quarternion.quaternion_to_mat(a)
    p = (1, -1, 1)
    print(np.matmul(Ra, p))
    print(a * Quarternion(p) * a.inv())
    print(Quarternion.rotate_with_quaternion(a, p))
