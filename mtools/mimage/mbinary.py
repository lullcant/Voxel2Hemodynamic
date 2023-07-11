#!/usr/bin/python3
# -*- coding:utf-8 -*-

###################################################################
## File: mpretreat.py
## Author: MiaoMiaoYang
## Created: 20.08.22
## Last Changed: 20.10.07
###################################################################

import numpy as np
import SimpleITK as sitk


## 在平面的二维的二值图像中找标签的边缘
def get_2D_label_contour(mask, inImage=True):
    '''
    在平面的二维的二值图像中寻找二值标签图像的内/外边缘
    :param mask:   标签图像，必须是二值图像
    :param inImage: 在标签内部找边缘还是在标签外部找边缘
    :return: array，图像
    '''
    import numpy as np
    from math import ceil, floor
    from skimage import measure

    mask = np.asarray(mask)
    assert np.unique(mask).shape[0] == 2, "input label is not a binary image in getContourIn2D"

    ## 新建一个空白图像
    blank = np.zeros(mask.shape, dtype=np.int)

    ## 把所有label的边缘找到，包含非拓扑结构
    contours = measure.find_contours(mask, 0.5)
    ## 遍历每一个拓扑结构的边缘
    for index, contour in enumerate(contours):
        ## 在图像内/外寻找边缘
        for n, dot in enumerate(contour):
            if mask[ceil(dot[0])][ceil(dot[1])] == inImage:
                blank[ceil(dot[0])][ceil(dot[1])] = 1
            if mask[ceil(dot[0])][floor(dot[1])] == inImage:
                blank[ceil(dot[0])][floor(dot[1])] = 1
            if mask[floor(dot[0])][ceil(dot[1])] == inImage:
                blank[floor(dot[0])][ceil(dot[1])] = 1
            if mask[floor(dot[0])][floor(dot[1])] == inImage:
                blank[floor(dot[0])][floor(dot[1])] = 1

    return blank


## 在立体的三维的二值图像中找标签的边缘，marching cubes algorithm
def get_3D_label_contour(mask, inImage=True):
    '''
    在立体的三维的二值图像中寻找二值标签图像的内/外边缘
    利用marching_cubes面重建方法实现，可能与预想的存在一些偏差
    :param mask:   标签图像，必须是二值图像
    :param inImage: 在标签内部找边缘还是在标签外部找边缘
    :return: array，图像
    '''
    import numpy as np
    from math import ceil, floor
    from skimage import measure

    mask = np.asarray(mask)
    assert np.unique(mask).shape[0] == 2, "input label is not a binary image in getContourIn3D"

    ## 新建一个空白图像
    blank = np.zeros(mask.shape, dtype=np.int)

    ## 把所有label的边缘找到，包含非拓扑结构
    verts, faces, normals, values = measure.marching_cubes_lewiner(mask, 0.5)

    ## 遍历每一个拓扑结构的边缘
    for index, dot in enumerate(verts):
        upx, dwx = ceil(dot[0]), floor(dot[0])
        upy, dwy = ceil(dot[1]), floor(dot[1])
        upz, dwz = ceil(dot[2]), floor(dot[2])

        if mask[upx][upy][upz] == inImage:
            blank[upx][upy][upz] = 1

        if mask[upx][upy][dwz] == inImage:
            blank[upx][upy][dwz] = 1

        if mask[upx][dwy][upz] == inImage:
            blank[upx][dwy][upz] = 1

        if mask[upx][dwy][dwz] == inImage:
            blank[upx][dwy][dwz] = 1

        if mask[dwx][upy][upz] == inImage:
            blank[dwx][upy][upz] = 1

        if mask[dwx][upy][dwz] == inImage:
            blank[dwx][upy][dwz] = 1

        if mask[dwx][dwy][upz] == inImage:
            blank[dwx][dwy][upz] = 1

        if mask[dwx][dwy][dwz] == inImage:
            blank[dwx][dwy][dwz] = 1

    return blank


## 求一幅标签图像的重心（中心点）
def get_label_center(mask):
    '''
    得到图片的重心点
    :param mask: numpy格式数据，得到图片的重心位置，可适应多维度
    :return: List，维度和label一样，得到每一个维度上的图片重心坐标[0为起始点]
    理论上可适应各种图片，但因为非标签图片像素值较大，运算时会出现溢出报错，所以这里只允许0,1标签图像
    '''
    import numpy as np
    mask = np.asarray(mask)

    assert len(np.unique(mask)) == 2, 'input image is not a label in ./mtool/mcompute.py !'

    ## 如果没有标注，图片全0则输出图片的中心点
    if mask.sum() == 0:
        return ((np.asarray(mask.shape) - 1) / 2).tolist()

    ## 中心点
    center = []
    ## 图片形状Shape
    shape = mask.shape

    for dim in range(len(shape)):
        ## 变换维度
        tmp_order = np.asarray(range(0, len(shape)))
        tmp_order[dim] = 0
        tmp_order[0] = dim
        tmp = mask.transpose(tmp_order)

        ## 将数组进行压缩只剩两维：[当前计算维度，剩余维度]
        ## 这样的话在接下来的计算中，每一行的在当前计算的维度值都是相同的
        tmp = tmp.reshape([shape[dim], -1])

        ## 计算中心点
        all = 0
        sum = 0
        for w, line in enumerate(tmp):
            ## 每一行的维度值w乘以像素值所赋予的权重line
            all += line.dot(w).sum()
            sum += line.sum()
        center.append(all / sum)

    return center


## 获得标签的最大的N个连通区域
def get_largest_n_connected_region(mask, n, background=0):
    '''
    将得到的标签求最大连通区域
    :param mask: 得到的标签数组
    :param background: 标签数组的背景，默认是0
    :return: largest_connected_region，numpy数组，只包含0,1标签
    因为用到skimage中的函数label，所以这里以mask指代标签
    '''
    from skimage.measure import label
    import numpy as np

    ## 返回每个连通区域，将每个连通区域赋予不同的像素值
    mask = label(mask, background=background)
    mask_flat = mask.flat

    ## bincount 标签中每个索引值的个数
    ## E.g. Array: [0, 1, 1, 3, 2, 1, 7]
    ## 遍历 0→7的索引，得到索引值得个数：[1, 3, 1, 1, 0, 0, 0, 1]，索引0出现了1次，索引1出现了3次...索引7出现了1次
    index_num = np.bincount(mask_flat)

    ## 将像素值出现的次数进行排序[从大到小]，选出非背景的像素值最多（最大连通）的像素值索引
    ## 而排序时产生的index其实就是像素值
    pixel_index = np.argsort(-index_num)
    pixel = pixel_index[0]

    connected_area = []
    for p in pixel_index:
        if p != background:
            tmp_area = np.zeros(mask.shape)
            tmp_area[mask == p] = 1
            tmp_area = np.asarray(tmp_area).astype(np.int)
            connected_area.append(tmp_area)
            if len(connected_area) == n:
                break

    return connected_area


## 获得标签的最大的N个连通区域 (使用 SimpleITK)
def get_largest_n_connected_region_sitk(mask, n=-1, background=0):
    ## https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958
    parts = sitk.ConnectedComponent(mask)
    sorts = sitk.RelabelComponent(parts, sortByObjectSize=True)
    num = len(np.unique(sorts)) -1
    num = num if n == -1 else min(n, num)

    connected_area = []
    for idx in range(num):
        connected_area.append(sorts == idx + 1)

    return connected_area



## 将多分类的标注分割成二值的多个标签
def get_separated_label(mask):
    import numpy as np
    mask = np.asarray(mask)
    ## class num
    num = np.unique(mask).shape[0]

    labels = []
    for cls in range(1, num):
        tmp = mask.copy()
        tmp[tmp < cls] = 0
        tmp[tmp >= cls] = 1
        labels.append(tmp)

    return labels


## 将数组二值化 - 大津
def get_binary_mask(array, mode='otsu', fore=None):
    '''
    将数组使用otsu进行二值化
    :param   array:  进行二值化的数组
    :param   fore:   前景的阈值，如果为None则不进行前景提取
    :return: binary: 得到的二值化数组
    '''
    if mode == 'otsu':
        ## 计算阈值
        from skimage.filters import threshold_otsu
        ## 将前景提取来
        tmp = array[array > fore] if fore != None else array
        thresh = threshold_otsu(tmp)
        ## 二值化
        binary = array > thresh
        return binary, thresh

    return None


## 标注骨骼化
class Skeleton:
    def __init__(self, mask=None):
        self.mask = np.asarray(mask)
        assert np.unique(mask).shape[0] == 2, "mask is not a binary image"
        self.skeleton = None
        self.sk_graph = None

    def get_skeleton(self):
        from skimage.morphology import skeletonize
        self.skeleton = skeletonize(self.mask)
        return self.skeleton

    def get_graph(self):
        from mtools.mdatastruc.mgraph import Graph
        if self.skeleton is None:
            self.get_skeleton()

        ## Generate the graph of the skeleton
        sk_point = np.argwhere(self.skeleton == 1).tolist()
        sk_graph = Graph(num_verts=len(sk_point), vertices=sk_point)
        for uindex, pnt in enumerate(sk_point):
            ## find 4/8 neighbor
            four_neighbor = [
                [pnt[0] - 1, pnt[1]],
                [pnt[0] + 1, pnt[1]],
                [pnt[0], pnt[1] - 1],
                [pnt[0], pnt[1] + 1]
            ]

            eight_neighbor = four_neighbor + [
                [pnt[0] - 1, pnt[1] - 1],
                [pnt[0] + 1, pnt[1] + 1],
                [pnt[0] + 1, pnt[1] - 1],
                [pnt[0] - 1, pnt[1] + 1]
            ]

            neighbors = eight_neighbor

            for neighor in neighbors:
                if neighor in sk_point:
                    vindex = sk_point.index(neighor)
                    sk_graph.add_edge(uindex, vindex)

        ## remove the circles of the skeleton graph
        self.sk_graph = sk_graph.get_acyclic_graph()
        return self.sk_graph

    def get_key_graph(self, threshold):
        if self.sk_graph is None:
            self.get_graph()

        while True:
            sk_graph = self.sk_graph.get_prunned_graph(threshold=threshold)
            if sk_graph.get_verts_num() == self.sk_graph.get_num_verts():
                break
            self.sk_graph = sk_graph

        self.sk_graph = self.sk_graph.get_key_graph()
        return self.sk_graph


def test_Skeleton():
    from scipy.ndimage import binary_fill_holes, binary_closing
    import matplotlib.pyplot as plt

    mipo = np.load('../support/mip.npy')
    plt.imshow(mipo, cmap=plt.cm.bone)
    plt.axis('off')
    plt.show()

    #################################################
    ## Binary Image
    #################################################
    hist, bins = np.histogram(mipo.ravel())
    threshold = bins[-3]
    binmip = mipo > threshold
    binmip = binary_closing(binmip)
    binmip = binary_fill_holes(binmip)
    binmip = get_largest_n_connected_region(binmip, 1)[0]
    plt.imshow(binmip, cmap=plt.cm.bone)
    plt.axis('off')
    plt.show()

    #################################################
    ## Sketelon
    #################################################
    sk = Skeleton(mask=binmip)
    sk_image = sk.get_skeleton()
    plt.imshow(sk_image, cmap=plt.cm.bone)
    plt.axis('off')
    plt.show()

    ## Graph
    sk_graph = sk.get_graph()
    sk_point = np.asarray(sk_graph.get_verts())
    plt.imshow(binmip, cmap=plt.cm.bone)
    plt.axis('off')
    plt.scatter(sk_point[:, 1], sk_point[:, 0], s=0.5, color='brown')
    plt.show()

    ## Key Graph
    sk_graph = sk.get_key_graph(threshold=20)
    plt.imshow(binmip, cmap=plt.cm.bone)
    plt.axis('off')
    plt.title('key graph')
    ## edges
    edges = sk_graph.get_edges(is_index=False)
    for edge in edges:
        left_point = edge[0]
        rigt_point = edge[1]
        x_point = [left_point[1], rigt_point[1]]
        y_point = [left_point[0], rigt_point[0]]
        plt.plot(x_point, y_point, linewidth=1, color='red')

    ## points
    ins = np.asarray(sk_graph.get_key_points(mode='ins', is_index=False))
    end = np.asarray(sk_graph.get_key_points(mode='end', is_index=False))
    plt.scatter(ins[:, 1], ins[:, 0], s=20, color='blue')
    plt.scatter(end[:, 1], end[:, 0], s=20, color='green')

    plt.show()


if __name__ == '__main__':
    test_Skeleton()
