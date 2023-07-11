#!/usr/bin/python3
# -*- coding:utf-8 -*-

###################################################################
## File: mpretreat.py
## Author: MiaoMiaoYang
## Created: 20.08.22
## Last Changed: 20.08.22
## Description: utils about boxes
###################################################################

## 得到mask的外接矩形
def get_bounding_box(image, mask, isSITK=True):
    '''
    得到mask的外接矩形
    :param image: 医学图像 <SimpleITK.SimpleITK.Image>
    :param mask:  图像标签 <SimpleITK.SimpleITK.Image>
    :return:  box <numpy.ndarry> [x_lower,x_upper,y_lower,y_upper,z_lower,z_upper]
    bbox: 这个顺序是和原始的SimpleITK一致的
    但是SimpleITK -> (GetArrayFromImage) -> numpy 会导致翻转
    需要将bounding box的维度进行改变： (0, 1, 2) -> (2, 1, 0)，
    '''
    from radiomics.imageoperations import checkMask
    import SimpleITK as sitk

    if not isinstance(image, sitk.Image):
        image = sitk.GetImageFromArray(image, isVector=False)
        mask = sitk.GetImageFromArray(mask, isVector=False)

    tmp, _ = checkMask(imageNode=image, maskNode=mask)

    if isSITK:
        box = tmp
    else:
        box = [*tmp[4:6], *tmp[2:4], *tmp[0:2]]
    return box


## 判断bounding box是否符合条件，如果符合返回True，否则False
def check_bounding_box(bounding_box, limit_size):
    x_l, x_u, y_l, y_u, z_l, z_u = bounding_box

    if x_l < 0 or y_l < 0 or z_l < 0:
        return False

    if x_u >= limit_size[0] or y_u >= limit_size[1] or z_u >= limit_size[2]:
        return False

    return True


## 找到bouding box的中心
def get_center_from_box(box):
    wl, wu, hl, hu, dl, du = box[0], box[1], box[2], box[3], box[4], box[5]
    return [(wu + wl) // 2, (hu + hl) // 2, (dl + du) // 2]


## 根据大小和中心恢复bouding box，如果有shape要求进行revise
def get_box_from_center(center, box_size, image_shape=None):
    '''
    :param center: shape[3]，三维图像的中心坐标
    :param box_size: bounding box size
    :param image_shape: 三维图像的大小限制，默认为None
    :return:
    '''
    lower = [box_size[i] // 2 for i in range(len(box_size))]
    center = [lower[i] if center[i] < lower[i] else center[i] for i in range(len(box_size))]

    if image_shape is not None:
        upper = [image_shape[i] - box_size[i] // 2 for i in range(len(box_size))]
        center = [upper[i] if center[i] > upper[i] else center[i] for i in range(len(box_size))]

    wu, hu, du = center[0] + box_size[0] // 2, center[1] + box_size[1] // 2, center[2] + box_size[2] // 2
    wl, hl, dl = wu - box_size[0] + 1, hu - box_size[1] + 1, du - box_size[2] + 1

    return [wl, wu, hl, hu, dl, du]


## 根据条件重新设置bounding_box的大小
def resize_bounding_box(bounding_box, limit_size, min_size=[64, 64, 64], force=False):
    '''
    根据条件重新设置bounding_box的大小，三个维度的大小限制遵循关系[axial,sagittal,coronal]，
    所以在图像裁剪时，相应的位置应该相反，E.g. Box[32,64,64]， Image Shape(Limit size) [512,512,78]
    :param bounding_box: 原来的bounding_box
    :param limit_size:   图像的尺寸，不能超过这个尺寸 image.shape，不能等于
    :param min_size:     重新设置的bounding_box的最小尺寸
    :param force:        是否强制按照min size去选取boudning box
    :return: 重新设置后的Bounding_box
    '''
    x_l, x_u, y_l, y_u, z_l, z_u = bounding_box

    def resize_dim(lower, upper, limit, mmin):
        center = (upper + lower) // 2

        if center < mmin // 2:
            return 0, mmin if force else max(mmin, upper)

        if center > limit - mmin // 2:
            return limit - mmin - 1 if force else min(limit - mmin - 1, lower), limit - 1

        return center - mmin // 2 if force else min(center - mmin // 2, lower), \
               center - mmin // 2 + mmin if force else max(center - mmin // 2 + mmin, upper)

    x_l, x_u = resize_dim(x_l, x_u, limit=limit_size[0], mmin=min_size[0])
    y_l, y_u = resize_dim(y_l, y_u, limit=limit_size[1], mmin=min_size[1])
    z_l, z_u = resize_dim(z_l, z_u, limit=limit_size[2], mmin=min_size[2])
    return [x_l, x_u, y_l, y_u, z_l, z_u]


## According to required bouding size and dicom image size to revise tumor box center
def revise_box_center(center, image_shape, require):
    '''
    :param center:  current box center
    :param image_shape:    image size，要求必须小于，没有等于条件
    :param require: required box size
    :return:
    '''

    assert (image_shape[0] > require[0]) and (image_shape[1] > require[1]) and (image_shape[2] > require[2])

    ## center的限制区域，center必须<upper,必须>=lower
    wl, hl, dl = require[0] // 2, require[1] // 2, require[2] // 2
    wu, hu, du = image_shape[0] - wl, image_shape[1] - hl, image_shape[2] - dl

    center[0] = (wu - 1) * (center[0] >= wu) + wl * (center[0] < wl) + center[0] * (center[0] >= wl) * (center[0] < wu)
    center[1] = (hu - 1) * (center[1] >= hu) + hl * (center[1] < hl) + center[1] * (center[1] >= hl) * (center[1] < hu)
    center[2] = (du - 1) * (center[2] >= du) + dl * (center[2] < dl) + center[2] * (center[2] >= dl) * (center[2] < du)

    return center


## 将一块区域按照bounding box的大小和设定的gap进行划分
def get_split_boxes_center_from_range(limit_range, box_size, gap):
    '''
    :param limit_range: 限制的区域
    :param gap:         相邻之间的间隔
    :return: boxes center: 划分bounding box的中心点
    '''
    import numpy as np
    box_size = np.asarray(box_size)

    assert limit_range[1] - limit_range[0] + 1 >= box_size[0], "limit range < bounding box in dim x"
    assert limit_range[3] - limit_range[2] + 1 >= box_size[1], "limit range < bounding box in dim y"
    assert limit_range[5] - limit_range[4] + 1 >= box_size[2], "limit range < bounding box in dim z"

    x = [i for i in range(limit_range[0] + box_size[0] // 2, limit_range[1] - box_size[0] // 2, gap[0])] \
        + [limit_range[1] - box_size[0] // 2]
    y = [i for i in range(limit_range[2] + box_size[1] // 2, limit_range[3] - box_size[1] // 2, gap[1])] \
        + [limit_range[3] - box_size[1] // 2]
    z = [i for i in range(limit_range[4] + box_size[2] // 2, limit_range[5] - box_size[2] // 2, gap[2])] \
        + [limit_range[5] - box_size[2] // 2]
    x, y, z = list(set(x)), list(set(y)), list(set(z))

    X, Y, Z = np.meshgrid(x, y, z)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    centers = [[X[index], Y[index], Z[index]] for index in range(len(X))]
    return centers


## 3D图片随机裁剪
def get_random_crop_boxes(shape, size, num):
    '''
    对3D图像和标签进行随机裁剪
    :param shape: 3D图片的shape (depth, hight, width)
    :param size:  crop的patch的大小 (64,64,64)
    :param num:   patch的数量
    :return: patchs = [(x1,y1,z1), ...] 每个patch的起始点
    '''
    import numpy as np
    shape = np.asarray(shape)
    size = np.asarray(size)

    assert len(shape) == 3, "input shape is not a 3D shape in randomCrop"
    assert len(size) == 3, "input size is not a 3D shape in randomCrop"
    assert (shape >= size).all(), "shape < size in randomCrop"

    ## 可以随机的范围
    scope = shape - size + 1
    d = np.random.randint(low=0, high=scope[0], size=num)
    h = np.random.randint(low=0, high=scope[1], size=num)
    w = np.random.randint(low=0, high=scope[2], size=num)
    patchs = np.stack([d, h, w], axis=1)
    return patchs


## 得到bbox的IOU
def get_bbox_IOU(c1, c2, shape):
    '''
    得到两个bbox的IOU
    :param c1: bbox1 的中心点
    :param c2: bbox2 的中心点
    :param shape: bbox的大小
    :return: 交并比
    '''
    common = abs(c1[0] - c2[0]) * \
             abs(c1[1] - c2[1]) * \
             abs(c1[2] - c2[2])
    total = shape[0] * shape[1] * shape[2] * 2 - common
    # print(total)
    return (common + 1) / (total + 1)


## 通过mask的外接矩形对图像和标签进行切割
def get_box_cropped_image_mask_sitk(image, mask, bounding_box):
    '''
    通过mask的外接矩形对图像和标签进行切割
    :param image: 医学图像 <SimpleITK.SimpleITK.Image>
    :param mask:  图像标签 <SimpleITK.SimpleITK.Image>
    :param bounding_box: <numpy.ndarry> [x_lower,x_upper,y_lower,y_upper,z_lower,z_upper]
    :return: image, mask
    '''
    from radiomics.imageoperations import cropToTumorMask
    image, mask = cropToTumorMask(image, mask, bounding_box)
    return image, mask


## 通过mask的外接矩形对图像和标签进行切割
def get_box_cropped_image_mask(image, bounding_box, DEBUG=False):
    '''
    通过mask的外接矩形对图像和标签进行切割
    :param image: 医学图像 <numpy.ndarry>
    :param bounding_box: <numpy.ndarry> [x_lower,x_upper,y_lower,y_upper,z_lower,z_upper]
    :return: image, mask
    '''
    import numpy as np

    image = np.asarray(image)
    shape = list(image.shape)

    lower = [bounding_box[2 * i] for i in range(3)]
    upper = [bounding_box[2 * i + 1] for i in range(3)]

    assert upper < shape, "the box doesn't match the image shape: (image shape: {} upper: {} lower: {})". \
        format(shape, upper, lower)
    assert lower >= [0] * 3, "the box doesn't match the image shape: (image shape: {} upper: {} lower: {})" \
        .format(shape, upper, lower)

    image = image[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    return image


def test_get_split_boxes_center_from_range():
    import numpy as np
    pancreas_bounding = [0, 31, 32, 63, 120, 152]
    gap = [16, 16, 16]
    bounding_box = [32, 32, 32]

    center = get_split_boxes_center_from_range(limit_range=pancreas_bounding, gap=gap, box_size=bounding_box)
    center = center[0]

    box = get_box_from_center(center, box_size=bounding_box)

    image = np.zeros([64, 512, 512])
    image = image[box[0]:box[1], box[2]:box[3], box[4]:box[5]]

    print(center)
    print(box)
    print(image.shape)


def test_get_bounding_box():
    import SimpleITK as sitk
    from mtool.mio import get_medical_image
    image_path = "../../data/Changhai/image/bai yu zhen.nrrd"
    organ_path = "../../data/Changhai/organ/bai yu zhen.nrrd"

    image = sitk.ReadImage(image_path)
    organ = sitk.ReadImage(organ_path)

    image, o, s, d, t = get_medical_image(image)
    print(type(image))
    exit()

    print(type(image))

    box = get_bounding_box(image, organ)

    from mtool.mio import get_medical_image, save_medical_image
    import numpy as np
    image, o, s, d, t = get_medical_image(image_path)
    organ, _, _, _, _ = get_medical_image(organ_path)

    result_image = np.zeros(image.shape)
    result_organ = np.zeros(image.shape)
    result_image[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = image[box[0]:box[1], box[2]:box[3], box[4]:box[5]]
    result_organ[box[0]:box[1], box[2]:box[3], box[4]:box[5]] = organ[box[0]:box[1], box[2]:box[3], box[4]:box[5]]

    save_medical_image(result_image, '../../result_image.nrrd', o, s, d)
    save_medical_image(result_organ, '../../result_organ.nrrd', o, s, d)


if __name__ == '__main__':
    test_get_bounding_box()
    # test_get_split_boxes_center_from_range()
