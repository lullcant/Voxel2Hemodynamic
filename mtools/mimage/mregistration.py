import SimpleITK as sitk


def get_registration_image(fix_image_path, moving_image_path, dst_path, mode):
    assert mode in ['rigid', 'affine', 'non-rigid'], "cannot  indentify the registration mode"

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(sitk.ReadImage(fix_image_path))
    elastixImageFilter.SetMovingImage(sitk.ReadImage(moving_image_path))

    if mode in ['rigid', 'affine']:
        param = sitk.GetDefaultParameterMap(mode)
    elif mode in ['non-rigid']:
        param = sitk.VectorOfParameterMap()
        param.append(sitk.GetDefaultParameterMap("affine"))
        param.append(sitk.GetDefaultParameterMap("bspline"))

    elastixImageFilter.SetParameterMap(param)
    elastixImageFilter.Execute()
    sitk.WriteImage(elastixImageFilter.GetResultImage(), dst_path)
