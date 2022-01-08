import numpy as np
from matplotlib import pyplot as plt

__all__ = ["Point3D", "list2mat", "mat2list"]


class Point3D(object):
    def __init__(self, vector, color=None):
        super(Point3D, self).__init__()
        self.color = color
        if isinstance(vector, Point3D):
            self.p = vector.p
        else:
            vec = np.array(vector).reshape((-1,))
            if vec.shape[0] == 4:
                self.ph = np.array(vec)
            elif vec.shape[0] == 3:
                self.p = np.array(vec)

    def __repr__(self):
        return str(self.p)

    def __setattr__(self, key, value):
        if key == 'x':
            self.__dict__[key] = value
            self.__dict__['p'][0] = value
            self.__dict__['ph'][0] = value
        elif key == 'y':
            self.__dict__[key] = value
            self.__dict__['p'][1] = value
            self.__dict__['ph'][1] = value
        elif key == 'z':
            self.__dict__[key] = value
            self.__dict__['p'][2] = value
            self.__dict__['ph'][2] = value
        elif key == 'p':
            self.__dict__[key] = np.array(value)
            self.__dict__['x'] = value[0]
            self.__dict__['y'] = value[1]
            self.__dict__['z'] = value[2]
            ph = np.ones((4,))
            ph[0:3] = value
            self.__dict__['ph'] = ph
        elif key == 'ph':
            self.__dict__[key] = np.array(value)
            self.__dict__['p'] = value[0:3]
            self.__dict__['x'] = value[0]
            self.__dict__['y'] = value[1]
            self.__dict__['z'] = value[2]
        elif key == "color":
            assert value is None or len(value) == 3, "color should be None or rgb"
            self.__dict__["color"] = value
        else:
            raise KeyError

    def __sub__(self, b):
        if isinstance(b, Point3D):
            return self.p - b.p     # Point sub a Point result to vector
        else:
            return Point3D(self.p - np.array(b))    # get the src Point3D of the vector

    def __add__(self, b):
        if isinstance(b, Point3D):
            return Point3D(self.p + b.p)
        else:
            return Point3D(self.p + np.array(b))    # Point add a vector result to another Point3D

    def __mul__(self, b):
        if isinstance(b, Point3D):
            return np.matmul(self.p, b.p)   # dot product
        else:
            return Point3D(self.p * b)  # Point * scalar

    def __truediv__(self, other):
        return Point3D(self.p / other)  # Point / scalar

    def rigid_transform(self, *args):
        if len(args) == 1:
            T = args[0]
            q = Point3D(np.matmul(T, self.ph))
        elif len(args) == 2:
            R = args[0]
            t = args[1]
            q = Point3D(np.matmul(R, self.p)) + t
        return q

    def plot3d(self, ax, marker='o', color=None, s=40):
        color = color or self.color or "blue"
        ax.scatter(self.x, self.y, self.z, s=s, marker=marker, color=color)


def list2mat(plist):
    N = len(plist)
    mat = np.zeros((N, 3), np.float32)
    for i in range(N):
        mat[i, :] = plist[i].p
    return mat


def mat2list(mat):
    N = mat.shape[0]
    plist = []
    for i in range(N):
        plist.append(Point3D(mat[i]))
    return plist


if __name__ == "__main__":
    R = np.eye(3)
    t = [1, 0, 2]
    p1 = Point3D((1, 1, 1))
    p2 = p1.rigid_transform(R, t)
    print(p1, p2)

    v1 = p1 - p2
    print(v1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.show()
