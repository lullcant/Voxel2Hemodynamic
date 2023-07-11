import matplotlib.pyplot as plt
from scipy.special import comb
from scipy import interpolate
import numpy as np
import math


class Line:
    def __init__(self, Points, InterpolationNum):
        self.demension = Points.shape[1]  # 点的维数
        self.segmentNum = InterpolationNum - 1  # 段数
        self.num = InterpolationNum  # 单段插补(点)数
        self.pointsNum = Points.shape[0]  # 点的个数
        self.Points = Points  # 所有点信息

    def getLinePoints(self):
        # 每一段的插补点
        pis = np.array(self.Points[0])
        # i是当前段
        for i in range(0, self.pointsNum - 1):
            sp = self.Points[i]
            ep = self.Points[i + 1]
            dp = (ep - sp) / (self.segmentNum)  # 当前段每个维度最小位移
            for i in range(1, self.num):
                pi = sp + i * dp
                pis = np.vstack((pis, pi))
        return pis


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def get_bessel_spline(points, N=1000):
    """
    https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    zPoints = np.array([p[2] for p in points])

    t = np.linspace(0.0, 1.0, N)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.flip(np.dot(xPoints, polynomial_array))
    yvals = np.flip(np.dot(yPoints, polynomial_array))
    zvals = np.flip(np.dot(zPoints, polynomial_array))

    return np.concatenate([[xvals], [yvals], [zvals]], axis=0).transpose(1, 0)


def get_b_spline_3d(points, N):
    ## points 不能存在相同的点
    points = np.asarray(points).copy()
    _, index, counts = np.unique(points, axis=0, return_index=True, return_counts=True)
    index = list(set([i for i in range(len(points))]) - set(index))
    points = np.delete(points, index, axis=0)

    if len(points) <=3:
        points = get_bessel_spline(points,N=6)


    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    tck, u = interpolate.splprep([x, y, z], k=3, s=0)
    u_fine = np.linspace(0, 1, N, endpoint=True)
    bspline = interpolate.splev(u_fine, tck)
    bspline = np.asarray(bspline).transpose(1, 0)
    return bspline


def get_b_spline_2d(points, N):
    points = np.asarray(points).copy()
    x = points[:, 0]
    y = points[:, 1]
    tck, u = interpolate.splprep([x, y], k=3, s=0)
    u_fine = np.linspace(0, 1, N, endpoint=True)
    bspline = interpolate.splev(u_fine, tck)
    bspline = np.asarray(bspline).transpose(1, 0)
    return bspline


def test3d():
    points = np.array([
        [1, 3, 0], [1.5, 1, 0], [4, 2, 0],
        [4, 3, 4], [2, 3, 11], [5, 5, 9]
    ])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # 标记控制点
    for i in range(0, points.shape[0]):
        ax.scatter(points[i][0], points[i][1], points[i][2], marker='o', color='black')
        ax.text(points[i][0], points[i][1], points[i][2], i, size=12)

    # 直线连接控制点
    l = Line(points, 1000)
    pl = l.getLinePoints()
    ax.plot3D(pl[:, 0], pl[:, 1], pl[:, 2], color='k')

    # # 贝塞尔曲线连接控制点
    matpi = get_bessel_spline(points.copy(), 1000)
    ax.plot3D(matpi[:, 0], matpi[:, 1], matpi[:, 2], color='r')

    ax.plot3D(matpi[:200, 0], matpi[:200, 1], matpi[:200, 2])
    ax.plot3D(matpi[200:400, 0], matpi[200:400, 1], matpi[200:400, 2])
    ax.plot3D(matpi[400:600, 0], matpi[400:600, 1], matpi[400:600, 2])
    ax.plot3D(matpi[600:800, 0], matpi[600:800, 1], matpi[600:800, 2])
    ax.plot3D(matpi[800:, 0], matpi[800:, 1], matpi[800:, 2])

    # B曲线连接控制点
    matpi = get_b_spline_3d(points.copy(), 1000)
    ax.plot3D(matpi[:, 0], matpi[:, 1], matpi[:, 2], color='b')

    plt.show()


def test2d():
    import numpy as np
    import scipy.interpolate as interpolate
    import matplotlib.pyplot as plt

    x = np.array([0., 1.2, 1.9, 3.2, 4., 6.5])
    y = np.array([0., 2.3, 3., 4.3, 2.9, 3.1])

    points = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    spline = get_b_spline_2d(points, N=100)

    plt.plot(x, y)
    plt.plot(spline[:, 0], spline[:, 1])
    plt.show()


if __name__ == '__main__':
    # test2d()
    # exit()
    test3d()
