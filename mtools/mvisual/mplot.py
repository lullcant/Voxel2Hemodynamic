import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_gaussian(mu, cov, radius=5, alpha=1.0):
    '''
    :param mu:
    :param cov:
    :param radius: 画图半径默认是5
    :param alpha: 透明度
    :param zorder: 等高线几等分
    :return:
    '''
    from scipy.stats import multivariate_normal

    ## 设定1000个采样点
    sample_num = 100 * radius
    X, Y = np.meshgrid(np.linspace(mu[0] - radius, mu[0] + radius, sample_num),
                       np.linspace(mu[1] - radius, mu[1] + radius, sample_num))
    Z = multivariate_normal(mean=mu, cov=cov).pdf(np.dstack([X, Y])).reshape(sample_num, sample_num)

    ## zorder: 画布是第基层
    plt.contourf(X, Y, Z, alpha=alpha, cmap=plt.cm.hot, zorder=-10)
    C = plt.contour(X, Y, Z, colors='black', alpha=alpha, zorder=-10)
    # 绘制等高线标签
    plt.clabel(C, inline=True, fontsize=8)


def plot_3d_gaussian(mu, cov, radius=5):
    from scipy.stats import multivariate_normal

    ## 设定1000个采样点
    sample_num = 100 * radius

    X, Y = np.meshgrid(np.linspace(mu[0] - radius, mu[0] + radius, sample_num),
                       np.linspace(mu[1] - radius, mu[1] - radius, sample_num))
    Z = multivariate_normal(mean=mu, cov=cov).pdf(np.dstack([X, Y])).reshape(sample_num, sample_num)

    fig = plt.figure(figsize=(6, 4))
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='seismic', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_3d_points(points, ax=None, end=True, color=None, size=20):
    points = np.asarray(points)
    assert len(points.shape) == 2 and points.shape[1] == 3, "point shape doesn't match"
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)

    ax.scatter(X, Y, Z, c=color, s=size, )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if end:
        plt.show()
    else:
        return ax


def plot_cross_plane(point, normal, image, spacing, w=32, h=32, padding=0):
    xx, yy = np.meshgrid(range(math.floor(point[0]) - w // 2, math.floor(point[0]) + w // 2),
                         range(math.floor(point[1]) - h // 2, math.floor(point[1]) + h // 2))
    d = -np.asarray(point).dot(normal)
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    shape = np.asarray(image).shape
    cross_section = np.zeros((w, h))
    for ii in range(w):
        for jj in range(h):
            cz = int(math.floor(zz[ii][jj] / spacing[2]))
            cy = int(math.floor(yy[ii][jj] / spacing[1]))
            cx = int(math.floor(xx[ii][jj] / spacing[0]))

            if cz >= shape[0] or cz < 0 or \
                    cy >= shape[1] or cy < 0 or \
                    cx >= shape[2] or cx < 0:
                cross_section[ii][jj] = 0
                continue

            cross_section[ii][jj] = image[cz][cy][cx]

    plt.imshow(cross_section)
    plt.show()


def draw_surface(image, show_axis=True, rstride=1, cstride=1):
    '''
    画出三维形式的分布图
    :param image:      分布的2D矩阵
    :param show_axis:  是否显示坐标轴
    :param rstride:    行间距
    :param cstride:    列间距
    :return:
    '''
    image = np.asarray(image)

    assert len(image.shape) == 2, "image is not a 2D image"

    width, height = image.shape

    X = np.linspace(0, width, width)
    Y = np.linspace(0, height, height)
    X, Y = np.meshgrid(X, Y)

    fig = plt.figure()

    # 创建3d图形的两种方式
    # ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    # rstride:行之间的跨度  cstride:列之间的跨度
    # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
    # vmax和vmin  颜色的最大值和最小值
    ax.plot_surface(X, Y, image, rstride=rstride, cstride=cstride, cmap=plt.get_cmap('rainbow'))

    # zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
    # offset : 表示等高线图投射到指定页面的某个刻度
    ax.contourf(X, Y, image, zdir='z', offset=-2)

    # 设置图像z轴的显示范围，x、y轴设置方式相同
    ax.set_zlim(-2, 2)

    # 去掉坐标轴
    if not show_axis:
        plt.axis('off')
    plt.show()


def plot_2d_radius(radius, edge_color='blue'):
    num = len(radius)
    circle = []
    for angle, R in zip(range(0, 360, 360 // num), radius):
        circle.append([math.cos(angle / 360 * 2 * math.pi) * R, math.sin(angle / 360 * 2 * math.pi) * R])

    circle.append(circle[0])
    circle = np.asarray(circle)
    plt.scatter(circle[:, 0], circle[:, 1], c='r', s=10)
    plt.plot(circle[:, 0], circle[:, 1], c=edge_color)
    # plt.show()
