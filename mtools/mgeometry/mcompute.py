import numpy as np
import math


## 将向量投影到一个平面上
def project_vector_on_plane(vector, normal):
    '''
    :param vector: 向量
    :param normal: 平面的法向
    :return: 投影后的向量
    '''

    bias = (np.dot(vector, normal) / np.linalg.norm(normal, keepdims=True) ** 2) * normal
    return vector - bias


def project_vector_on_vector(v1, v2):
    '''
    将向量v1投影到向量v2上
    :param v1:
    :param v2:
    :return:
    '''
    return (np.dot(v1, v2) / np.linalg.norm(v2, keepdims=True) ** 2) * v2


def test_project_line_on_plane():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vector = np.asarray([2, 5, 30])
    normal = np.asarray([1, 1, 7])
    prject = project_vector_on_plane(vector, normal)

    ## plot plane
    w = 32
    h = 32
    xx, yy = np.meshgrid(range(math.floor(vector[0]) - w // 2, math.floor(vector[0]) + w // 2),
                         range(math.floor(vector[1]) - h // 2, math.floor(vector[1]) + h // 2))
    d = -np.asarray(vector).dot(normal)
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(xx, yy, zz, alpha=0.8)
    plt.plot([0, vector[0] * 2], [0, vector[1] * 2], [0, vector[2] * 2])
    plt.plot([vector[0], vector[0] + prject[0]], [vector[1], vector[1] + prject[1]], [vector[2], vector[2] + prject[2]])

    prject = project_vector_on_vector(v1=vector, v2=normal)
    plt.plot([vector[0], vector[0] + prject[0]], [vector[1], vector[1] + prject[1]], [vector[2], vector[2] + prject[2]])
    plt.plot([vector[0], vector[0] + normal[0]], [vector[1], vector[1] + normal[1]], [vector[2], vector[2] + normal[2]])

    plt.show()


if __name__ == '__main__':
    test_project_line_on_plane()
