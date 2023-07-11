#!/usr/bin/python3
# -*- coding:utf-8 -*-

###################################################################
## Author: MiaoMiaoYang
## Created: 19.08.29
## Last Changed: 20.11.17
###################################################################


import random

from scipy.ndimage.interpolation import shift, zoom

from mtools.mimage.mimage3d import get_3d_rotate, get_3d_interpolation


## 对图像进行数据增强（旋转）
def get_augment_data_2d(images, angle=None, shifts=None, scales=None, bias=100):
    '''
    对图像进行数据增强
    :param images: list, contain many images to do the same operation (ratate, shift, scale)
    :param angle: rotate angle range in three dim
    :param shifts: shift deviation range in three dim
    :param scales: scale range
    :return:
    '''

    if angle is not None:
        angle = random.uniform(-angle, angle)

    if shifts is not None:
        x_shift = random.uniform(-shifts[0], shifts[0])
        y_shift = random.uniform(-shifts[1], shifts[1])

    results = []
    for image in images:
        mmin = -image.min()
        image = image + mmin


        if angle is not None:
            image = get_3d_rotate(image=image, angle=x_angle, plane='x')
            image = get_3d_rotate(image=image, angle=y_angle, plane='y')
            image = get_3d_rotate(image=image, angle=z_angle, plane='z')
            image[image < bias] = 0

        if shifts is not None:
            image = shift(input=image, shift=[x_shift, y_shift, z_shift])
            image[image < 0] = 0

        image = image - mmin
        results.append(image)

    return results


## 对图像进行数据增强（旋转）
def get_augment_data_3d(images, angles=None, shifts=None, scales=None, bias=100):
    '''
    对图像进行数据增强
    :param images: list, contain many images to do the same operation (ratate, shift, scale)
    :param angles: rotate angle range in three dim
    :param shifts: shift deviation range in three dim
    :param scales: scale range
    :return:
    '''

    if angles is not None:
        assert list(angles) > [0] * 3, 'angle <= 0 in get augment data'
        x_angle = random.uniform(-angles[0], angles[0])
        y_angle = random.uniform(-angles[1], angles[1])
        z_angle = random.uniform(-angles[2], angles[2])

    if shifts is not None:
        assert list(shifts) > [0] * 3, 'shift <= 0 in get augment data'
        x_shift = random.uniform(-shifts[0], shifts[0])
        y_shift = random.uniform(-shifts[1], shifts[1])
        z_shift = random.uniform(-shifts[2], shifts[2])

    if scales is not None:
        assert len(scales) == 2, "scale range is incorrect"
        scale = random.uniform(scales[0], scales[1])

    # print("angle:{} shift:{} scale:{}".format([x_angle, y_angle, z_angle] if angles is not None else "None",
    #                                           [x_shift, y_shift, z_shift] if shifts is not None else "None",
    #                                           scale if scales is not None else "None"))

    results = []
    for image in images:
        mmin = -image.min()
        image = image + mmin

        if scales is not None:
            shape = list(image.shape)
            image = zoom(image, zoom=0.5)
            image = get_3d_interpolation(image, dst_size=shape)
            image[image < 0] = 0

        if angles is not None:
            image = get_3d_rotate(image=image, angle=x_angle, plane='x')
            image = get_3d_rotate(image=image, angle=y_angle, plane='y')
            image = get_3d_rotate(image=image, angle=z_angle, plane='z')
            image[image < bias] = 0

        if shifts is not None:
            image = shift(input=image, shift=[x_shift, y_shift, z_shift])
            image[image < 0] = 0

        image = image - mmin
        results.append(image)

    return results


def test_get_augment_data():
    from mtools.mio import get_medical_image, save_medical_image
    import numpy as np
    import os
    cbct, o, s, d, t = get_medical_image("../../data/matched/cbct/ye wei dong-47.nrrd")
    mict, _, _, _, _ = get_medical_image(
        "../../data/matched/microCT-registration/rigid-microCT-without_bg/ye wei dong-47.nrrd")
    anno, _, _, _, _ = get_medical_image(
        "../../data/matched/microCT-registration/rigid-microCT-label/ye wei dong-47.nrrd")

    test_dire = '../../test'
    os.makedirs(test_dire, exist_ok=True)

    images = [cbct, mict, anno * 200]

    angles = get_augment_data_3d(images, angles=(90, 90, 90), shifts=None, scales=None)
    save_medical_image(angles[0], os.path.join(test_dire, 'angle-cbct.nrrd'), o, s, d, t)
    save_medical_image(angles[1], os.path.join(test_dire, 'angle-mict.nrrd'), o, s, d, t)
    save_medical_image(np.asarray(angles[2] > 100).astype(np.int), os.path.join(test_dire, 'angle-anno.nrrd'), o, s, d,
                       t)

    angles = get_augment_data_3d(images, angles=None, shifts=(50, 50, 50), scales=None)
    save_medical_image(angles[0], os.path.join(test_dire, 'shift-cbct.nrrd'), o, s, d, t)
    save_medical_image(angles[1], os.path.join(test_dire, 'shift-mict.nrrd'), o, s, d, t)
    save_medical_image(np.asarray(angles[2] > 100).astype(np.int), os.path.join(test_dire, 'shift-anno.nrrd'), o, s, d,
                       t)

    angles = get_augment_data_3d(images, angles=None, shifts=None, scales=(0.2, 2))
    save_medical_image(angles[0], os.path.join(test_dire, 'scale-cbct.nrrd'), o, s, d, t)
    save_medical_image(angles[1], os.path.join(test_dire, 'scale-mict.nrrd'), o, s, d, t)
    save_medical_image(np.asarray(angles[2] > 0.1).astype(np.int), os.path.join(test_dire, 'scale-anno.nrrd'), o, s, d,
                       t)


if __name__ == "__main__":
    test_get_augment_data()
