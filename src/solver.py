import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from point import *
from camera import *
from geometry import *


# geometric operations with tensorflow
def tf_cross_mat(k):
    zero = tf.constant(0.0, dtype=tf.float32, shape=(1, 1), name="zero")
    k0 = tf.reshape(k[0], (1, -1))
    k1 = tf.reshape(k[1], (1, -1))
    k2 = tf.reshape(k[2], (1, -1))
    line0 = tf.concat([zero, -k2, k1], axis=1)
    line1 = tf.concat([k2, zero, -k0], axis=1)
    line2 = tf.concat([-k1, k0, zero], axis=1)
    K = tf.concat([line0, line1, line2], axis=0)
    return K


def tf_rotation_mat(axis, theta):
    I = tf.constant(np.eye(3), dtype=tf.float32, shape=(3, 3), name="I3")
    K = tf_cross_mat(axis)
    return I + tf.sin(theta) * K + (1 - tf.cos(theta)) * tf.matmul(K, K)


def tf_N_mat(n):
    I = tf.constant(np.eye(3), dtype=tf.float32, shape=(3, 3), name="I3")
    return I - tf.matmul(n, tf.transpose(n))


def tf_intrinsic_mat(sx, sy, imgW, imgH):
    zero = tf.constant(0.0, tf.float32, (1, 1), name="zero")
    one = tf.constant(1.0, tf.float32, (1, 1), name="one")
    K0 = tf.concat([-sx, zero, imgH * sx / 2], axis=1)
    K1 = tf.concat([zero, sy, -imgW * sy / 2], axis=1)
    K2 = tf.concat([zero, zero, one], axis=1)
    return tf.concat([K0, K1, K2], axis=0)


def quad_form(n1, Q, n2):
    return tf.matmul(tf.matmul(tf.reshape(n1, (1, -1)), Q), tf.reshape(n2, (3, -1)))


def test_regression(Data, Label, epoch=1):
    tf.reset_default_graph()
    data = tf.placeholder(tf.float32, (3, None))
    label = tf.placeholder(tf.float32, (3, None))

    # variables & graph
    axis = tf.get_variable("axis", (3, 1), tf.float32, initializer=tf.constant_initializer((0, 0, 1)), trainable=True)
    theta = tf.get_variable("theta", (1, 1), tf.float32, initializer=tf.constant_initializer(0.5), trainable=True)
    R = tf_rotation_mat(axis, theta)
    T = tf.get_variable("T", (3, 1), tf.float32, initializer=tf.constant_initializer(np.zeros((3, 1))), trainable=True)

    out = tf.add(tf.matmul(R, data), T)

    # loss
    loss = tf.losses.mean_squared_error(label, out)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.05, global_step, decay_steps=10, decay_rate=0.99, staircase=False)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for iter in range(epoch):
            for i in range(Data.shape[1]):
                _, loss_, lr, step = sess.run([train, loss, learning_rate, global_step],
                                          feed_dict={
                                            data: Data[:, i].reshape(3, -1),
                                            label: Label[:, i].reshape(3, -1)
                                          })
                print("step %d: loss = %f, lr = %f" % (step, loss_, lr))

        print("theta =", theta.eval())
        print("axis =", axis.eval())
        return R.eval(), T.eval()


def tf_decomp_essential_mat(E, lr=1e-3, epoch=1):
    tf.reset_default_graph()
    E_tensor = tf.placeholder(tf.float32, (3, 3))

    # variables
    axis = tf.get_variable("axis", (3, 1), tf.float32, initializer=tf.constant_initializer((1, 0, 1)), trainable=True)
    theta = tf.get_variable("theta", (1, 1), tf.float32, initializer=tf.constant_initializer(np.pi / 4), trainable=True)
    R = tf_rotation_mat(axis, theta)
    T = tf.get_variable("T", (3, 1), tf.float32, initializer=tf.constant_initializer(np.ones((3, 1))), trainable=True)
    E_ = tf.matmul(tf_cross_mat(T), R)

    # loss
    loss = tf.reduce_mean(tf.square(E_ - E_tensor))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=10, decay_rate=0.99, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for iter in range(epoch):
            _, loss_, lr, step = sess.run([train, loss, learning_rate, global_step], feed_dict={E_tensor:E})
            if iter % 100 == 0:
                print("step %d: loss = %f, lr = %f" % (step, loss_, lr))

        print("theta = {}\n, axis = \n{}\n".format(theta.eval(), axis.eval()))
        return R.eval(), T.eval()


def tf_decomp_fundamental_mat(input_F, lr=1e-3, epoch=1):
    tf.reset_default_graph()
    F_tensor = tf.placeholder(tf.float32, (3, 3))

    # variables
    sx = tf.get_variable("sx", (1, 1), tf.float32, initializer=tf.constant_initializer(0.01), trainable=True)
    sy = sx
    axis = tf.get_variable("axis", (3, 1), tf.float32, initializer=tf.constant_initializer((1, 0, 1)), trainable=True)
    theta = tf.get_variable("theta", (1, 1), tf.float32, initializer=tf.constant_initializer(np.pi / 4), trainable=True)
    R = tf_rotation_mat(axis, theta)
    T = tf.get_variable("T", (3, 1), tf.float32, initializer=tf.constant_initializer(np.ones((3, 1))), trainable=True)
    E = tf.matmul(tf_cross_mat(T), R)

    # loss
    loss = tf.reduce_mean(tf.square(E_ - E_tensor))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lr, global_step, decay_steps=10, decay_rate=0.99, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for iter in range(epoch):
            _, loss_, lr, step = sess.run([train, loss, learning_rate, global_step], feed_dict={E_tensor:E})
            if iter % 100 == 0:
                print("step %d: loss = %f, lr = %f" % (step, loss_, lr))

        print("theta = {}\n, axis = \n{}\n".format(theta.eval(), axis.eval()))
        return R.eval(), T.eval()


if __name__ == "__main__":
    f = 1.0
    sx = 0.002
    sy = 0.002
    sx = sx / f
    sy = sy / f
    W = 1920
    H = 1080
    theta = np.pi * 0.9
    axis = np.array([0., 1., 1.])
    R = rodriguez(axis, theta)
    t = np.array([0, -3, 6])
    E = np.matmul(cross_mat(t), R)
    K = np.array([[-sx, 0.0, H / 2 * sx],
                  [0.0, sy, -W / 2 * sy],
                  [0.0, 0.0, 1.0]])
    K_ = np.linalg.pinv(K)

    p2d1 = np.fromfile("../Data/p2d1.dat", np.float64).reshape((-1, 2))
    p2d2 = np.fromfile("../Data/p2d2.dat", np.float64).reshape((-1, 2))
    p2d1 = np.column_stack((p2d1, np.ones((p2d1.shape[0], 1))))
    p2d2 = np.column_stack((p2d2, np.ones((p2d2.shape[0], 1))))
    print(p2d1)
    print(p2d2)
    F = ransac_f_mat(p2d1, p2d2, eps=1e-3)

    E = np.matmul(np.matmul(K_.T, F), K_)
    R, t = decompose_essential_mat(E)
    R, t = tf_decomp_essential_mat(E, lr=0.1, epoch=2000)

    print(R)
    print(t)