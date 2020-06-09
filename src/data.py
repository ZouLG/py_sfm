import numpy as np
from point import Point3D, list2mat, mat2list


def generate_rand_points(num=1, loc=[0, 0, 0], scale=[1, 1, 1]):
    x = np.random.normal(loc[0], scale[0], (num,))
    y = np.random.normal(loc[1], scale[1], (num,))
    z = np.random.normal(loc[2], scale[2], (num,))
    points = []
    for i in range(num):
        points.append(Point3D((x[i], y[i], z[i])))
    return points

