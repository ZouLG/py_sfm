import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Point3D:
    def __init__(self, vector):
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
        # ptmp = np.zeros((3,))
        # if key == 'x':
        #     ptmp[0] = value
        # elif key == 'y':
        #     ptmp[1] = value
        # elif key == 'z':
        #     ptmp[2] = value
        # elif key == 'p':
        #     ptmp = value
        # else:
        #     print("Error: class Point3D named %s" % key)
        #     raise KeyError
        # self.__dict__['x'] = ptmp[0]
        # self.__dict__['y'] = ptmp[1]
        # self.__dict__['z'] = ptmp[2]
        # self.__dict__['p'] = ptmp
        # self.__dict__['ph'][0: 3] = ptmp

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
        else:
            raise KeyError

    def __sub__(self, b):
        if isinstance(b, Point3D):
            return self.p - b.p     # Point sub a Point result to vector
        else:
            return Point3D(self.p - np.array(b))    # get the src point of the vector

    def __add__(self, b):
        if isinstance(b, Point3D):
            return Point3D(self.p + b.p)
        else:
            return Point3D(self.p + np.array(b))    # Point add a vector result to another point

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

    def plot3d(self, ax, marker='o', color='blue', s=40):
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


def get_point_by_idx(points, idx):
    if isinstance(points, list):
        return [points[i] for i in idx]
    else:
        return points[idx, :]


def save_points_to_file(plist, file_name):
    mat = list2mat(plist)
    mat.tofile(file_name)


def read_points_from_file(file_name):
    mat = np.fromfile(file_name, np.float32).reshape((-1, 3))
    return mat2list(mat)


if __name__ == "__main__":
    R = np.eye(3)
    t = [1, 0, 2]
    p1 = Point3D((1, 1, 1))
    p2 = p1.rigid_transform(R, t)
    print(p1, p2)

    v1 = p1 - p2
    print(v1)

    plane1 = ImgPlane((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plane1.show(ax)
    plt.show()
