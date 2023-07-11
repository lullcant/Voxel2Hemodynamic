import numpy as np
import SimpleITK as sitk
from mtools.mimage.mboxex import get_bounding_box, get_center_from_box, get_box_cropped_image_mask_sitk, \
    get_box_from_center


def get_3d_interpolation_sitk(image, dst_space=None, dst_size=None):
    '''
    :param image:     医学图像 <SimpleITK.SimpleITK.Image>
    :param dst_space: 目标间距 <numpy.array>
    :param dst_size:  目标尺寸 <numpy.array>
    :return:
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetDefaultPixelValue(0)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())

    if dst_space is not None:
        dst_size = image.GetSize() * np.asarray(dst_space) / image.GetSpacing()
        dst_size = np.asarray(dst_size).astype(np.int)

    if dst_size is not None:
        dst_space = image.GetSpacing() / np.asarray(dst_size) * image.GetSize()
        dst_space = np.asarray(dst_space)

    resample.SetSize(dst_size.tolist())
    resample.SetOutputSpacing(dst_space.tolist())
    return resample.Execute(image)


## padding SITK
def get_3d_padding_sitk(image, dst_shape, pad_value):
    '''
    :param image:     医学图像 <SimpleITK.SimpleITK.Image>
    :param dst_space: 目标间距 <numpy.array>
    :param pad_value: 填充值
    :return:
    '''

    pad = [j - i for (i, j) in zip(image.GetSize(), dst_shape)]
    if pad == [0, 0, 0]:
        return image

    lowerbound = np.asarray(pad) // 2
    upperbound = np.asarray(pad) - lowerbound

    pad = sitk.ConstantPadImageFilter()
    pad.SetConstant(pad_value)
    pad.SetPadLowerBound(np.asarray(lowerbound).tolist())
    pad.SetPadUpperBound(np.asarray(upperbound).tolist())
    return pad.Execute(image)


## 对3D进行插值
def get_3d_interpolation(image, dst_size):
    '''
    :param image: numpy image
    :param dst_size: expected image shape
    :return:
    '''
    from scipy.interpolate import RegularGridInterpolator

    image = np.asarray(image)
    shape = image.shape

    px, py, pz = shape
    sx, sy, sz = dst_size
    # print("shape:{} px:{} py:{} pz:{} sx:{} sy:{} sz:{}".format(shape, px, py, pz, sx, sy, sz))

    x = np.linspace(1, sx, px)
    y = np.linspace(1, sy, py)
    z = np.linspace(1, sz, pz)

    fn = RegularGridInterpolator((x, y, z), image)

    x = np.linspace(1, sx, sx)
    y = np.linspace(1, sy, sy)
    z = np.linspace(1, sz, sz)

    x_pts, y_pts, z_pts = np.meshgrid(x, y, z, indexing='ij')
    pts = np.concatenate([x_pts.reshape(-1, 1), y_pts.reshape(-1, 1), z_pts.reshape(-1, 1)], axis=1)

    resuts = np.asarray(fn(pts)).reshape((sx, sy, sz))
    return resuts


## 对3D进行旋转
def get_3d_rotate(image, angle, plane='z', mode="nearest"):
    '''
    :param image: numpy image
    :param angle: (y_rotate_angle,x_rotate_angle)
    :return:
    '''

    from scipy.ndimage import rotate

    if plane == 'z':
        axes = (0, 1)
    elif plane == 'y':
        axes = (0, 2)
    elif plane == 'x':
        axes = (1, 2)
    else:
        raise Exception("No rotation plane specified")

    ## reshape = False keeps the shape of original image
    image = rotate(image, angle, axes=axes, reshape=False, mode=mode)
    return image


def test_get_3d_interpolation():
    from mtools.mio import get_medical_image, save_medical_image
    img, o, s, d, _ = get_medical_image("../../data/Changhai/image/cao xue e.nrrd")
    print(img.shape)
    s = (s[0], s[1], s[2] / (84.5 / 80))

    result = get_3d_interpolation(img, map(int, [84.5, img.shape[1], img.shape[2]]))
    print(result.shape)
    save_medical_image(result, "../../cao.nrrd", o, s, d)


def test_get_3d_rotate():
    from mtools.mio import get_medical_image, save_medical_image
    img, o, s, d, _ = get_medical_image("../../data/Changhai/image/bai yu zhen.nrrd")
    org, _, _, _, _ = get_medical_image("../../data/Changhai/organ/bai yu zhen.nrrd")

    img = get_3d_rotate(image=img, angle=45, plane='x')
    org = get_3d_rotate(image=org, angle=45, plane='x')
    org = np.asarray(org > 0.1).astype(np.int)

    save_medical_image(img, "../../bai-img.nrrd", o, s, d)
    save_medical_image(org, "../../bai-org.nrrd", o, s, d)


if __name__ == "__main__":
    image = np.random.random([2, 2, 2]) * 2000

    print(image.max())
    print(image.min())

    image = norm_zero_one(image, span=[0, 2400])
    print(image.max())
    print(image.min())

    image = norm_zero_one(image)
    print(image.max())
    print(image.min())

    # test_get_3d_rotate()
    # test_get_3d_interpolation()
