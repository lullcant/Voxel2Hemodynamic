#!/usr/bin/python3
# -*- coding:utf-8 -*-

###################################################################
## Author: MiaoMiaoYang
## Created: 19.08.29
## Last Changed: 19.11.05
###################################################################

###################################################################
## SimpleITK 官方教程
## https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/
## https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks
## https://simpleitk-prototype.readthedocs.io/en/latest/user_guide/plot_image.html

## origin, spacing, direction 解释
## https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/03_Image_Details.html

## dicom序列写
## https://itk.org/SimpleITKDoxygen/html/DicomSeriesReadModifyWrite_2DicomSeriesReadModifySeriesWrite_8py-example.html
###################################################################

import os
import shutil
import sys
from tempfile import TemporaryDirectory

import SimpleITK as sitk
import natsort
import numpy as np
from PIL import Image


## 得到一组dicom序列图像,废弃不用！
def get_dicom_image(dire):
    '''
    加载一组dicom序列图像
    :param dire: dicom序列所在的文件夹路径，E.g. "E:/Work/Database/Teeth/origin/1/"
    :return: (array,origin,spacing,direction)
    array:  图像数组
    origin: 三维图像坐标原点
    spacing: 三维图像坐标间距
    direction: 三维图像坐标方向
    注意：实际取出的数组不一定与MITK或其他可视化工具中的方向一致！
    可能会出现旋转\翻转等现象，这是由于dicom头文件中的origin,spacing,direction的信息导致的
    在使用时建议先用matplotlib.pyplot工具查看一下切片的方式是否异常，判断是否需要一定的预处理
    注意：实际DICOM第一张可能是定位图，同时取出会导致位置错乱
    '''

    assert os.path.exists(dire), "{} is not existed".format(dire)
    assert os.path.isdir(dire), "{} is not a directory".format(dire)

    ## 厚度不一样，则有一样定位图
    thickness = dict()
    files = get_files_name(dire)
    for index in range(len(files)):
        file = sitk.ReadImage(os.path.join(dire, files[index]))
        sthick = file.GetMetaData('0018|0050')
        if sthick in thickness:
            thickness[sthick].append(files[index])
        else:
            thickness[sthick] = [files[index]]

    thickness = sorted(thickness.items(), key=lambda x: len(x[1]), reverse=True)
    files = thickness[0][1]

    with TemporaryDirectory() as dirname:
        for index in range(len(files)):
            shutil.copyfile(src=os.path.join(dire, files[index]), dst=os.path.join(dirname, files[index]))

        ## 重新加载图片
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOn()  # 加载公开的元信息
        reader.LoadPrivateTagsOn()  # 加载私有的元信息

        series = reader.GetGDCMSeriesIDs(dirname)
        filesn = reader.GetGDCMSeriesFileNames(dirname, series[0])

        reader.SetFileNames(filesn)
        dcmimg = reader.Execute()

    array = sitk.GetArrayFromImage(dcmimg)
    origin = dcmimg.GetOrigin()  # x, y, z
    spacing = dcmimg.GetSpacing()  # x, y, z
    direction = dcmimg.GetDirection()
    image_type = dcmimg.GetPixelID()  ## 原图像每一个像素的类型
    return array, {'origin': origin, 'spacing': spacing, 'direction': direction, 'type': image_type}


## 得到一个文件下所有的dicom文件
def get_dicom_images(dire):
    '''
    :param dire: Dicom序列所在文件夹路径（在我们的实验中，该文件夹下有多个dcm序列，混合在一起）
    :return:
    '''
    # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dire)
    print('Directory:{} has {} series.'.format(dire, len(series_IDs)))

    images = []
    proper = []
    for index in range(len(series_IDs)):
        # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[0]代表的是第一个序列的ID
        # 如果不添加series_IDs[0]这个参数，则默认获取第一个序列的所有切片路径
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dire, series_IDs[index])

        # 新建一个ImageSeriesReader对象
        reader = sitk.ImageSeriesReader()

        # 通过之前获取到的序列的切片路径来读取该序列
        reader.SetFileNames(series_file_names)

        ## 更新序列的参数数据
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()

        # 获取该序列对应的3D图像
        image = reader.Execute()

        item = {}
        keys = reader.GetMetaDataKeys(0)
        for key in keys:
            item[key] = reader.GetMetaData(0, key)

        images.append(image)
        proper.append(item)

    return len(series_IDs), images, proper


## 得到2D/3D的医学图像(除.dcm序列图像)
def get_medical_image(path):
    '''
    加载一幅2D/3D医学图像(除.dcm序列图像)，支持格式：.nii, .nrrd, ...
    :param path: 医学图像的路径/SimpleITK.SimpleITK.Image
    :return:(array,origin,spacing,direction)
    array:  图像数组
    origin: 三维图像坐标原点
    spacing: 三维图像坐标间距
    direction: 三维图像坐标方向
    image_type: 图像像素的类型
    注意：实际取出的数组不一定与MITK或其他可视化工具中的方向一致！
    可能会出现旋转\翻转等现象，这是由于dicom头文件中的origin,spacing,direction的信息导致的
    在使用时建议先用matplotlib.pyplot工具查看一下切片的方式是否异常，判断是否需要一定的预处理
    '''

    if isinstance(path, sitk.Image):
        reader = path
    else:
        assert os.path.exists(path), "{} is not existed".format(path)
        assert os.path.isfile(path), "{} is not a file".format(path)
        reader = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(reader)
    spacing = reader.GetSpacing()  ## 间隔
    origin = reader.GetOrigin()  ## 原点
    direction = reader.GetDirection()  ## 方向
    image_type = reader.GetPixelID()  ## 原图像每一个像素的类型，
    return array, {'origin': origin, 'spacing': spacing, 'direction': direction, 'type': image_type}


## 加载一张普通格式图片 2D
def get_normal_image(path, size=None):
    '''
    加载一幅普通格式的2D图像，支持格式：.jpg, .jpeg, .tif ...
    :param path: 图像的路径
    :param size: 对图像进行指定大小
    :return: array: numpy格式
    '''
    image = Image.open(path)
    if size is not None:
        image = image.resize(size)
    image = np.asarray(image)
    return image


## 按顺序得到当前目录下，所有文件（包括文件夹）的名字
def get_files_name(dire):
    '''
    按顺序得到当前目录下，所有文件（包括文件夹）的名字
    :param dire: 文件夹目录
    :return:files[list]，当前目录下所有的文件（包括文件夹）的名字，顺序排列
    '''

    assert os.path.exists(dire), "{} is not existed".format(dire)
    assert os.path.isdir(dire), "{} is not a directory".format(dire)

    files = os.listdir(dire)
    files = natsort.natsorted(files)
    return files


## get json content as dict
def get_json(file):
    import json
    with open(file, 'r', encoding='utf-8') as f:
        dicts = json.load(f)
    return dicts


## 获取csv文件
def get_csv(file, delimiter=','):
    import csv
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        result = list(reader)
    return result


def get_yaml(file):
    import yaml
    with open(file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_pickle(file):
    import pickle
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_obj(path, no_normal=False):
    with open(path, 'r') as file:
        lines = [line.rstrip() for line in file]

    vertices, normals, faces = [], [], []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('vn '):
            normals.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

    mesh = dict()
    mesh['faces'] = np.vstack(faces)
    mesh['vertices'] = np.vstack(vertices)

    if (not no_normal) and (len(normals) > 0):
        assert len(normals) == len(vertices), 'ERROR: #vertices != #normals'
        mesh['normals'] = np.vstack(normals)

    return mesh


def source_import(file_path):
    import importlib
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


## 将numpy数组保存为3D医学图像格式，支持 .nii, .nrrd
def save_medical_image(array, target_path, param=None):
    '''
    将得到的数组保存为医学图像格式
    :param array: 想要保存的医学图像数组，为避免错误，这个函数只识别3D数组
    :param origin:读取原始数据中的原点
    :param space: 读取原始数据中的间隔
    :param direction: 读取原始数据中的方向
    :param target_path: 保存的文件路径，注意：一定要带后缀，E.g.,.nii,.nrrd SimpleITK会根据路径的后缀自动判断格式，填充相应信息
    :param type: 像素的储存格式
    :return: None 无返回值
    注意，因为MITK中会自动识别当前载入的医学图像文件是不是标签(label)【通过是否只有0,1两个值来判断】
    所以在导入的时候，MITK会要求label的文件格式为unsigned_short/unsigned_char型，否则会有warning
    '''

    if isinstance(array, sitk.Image):
        image = array
        sitk.WriteImage(image, target_path, True)
        return

    # assert len(np.asarray(array).shape) == 3, "array's shape is {}, it's not a 3D array".format(np.asarray(array).shape)

    ## if isVector is true, then a 3D array will be treaded as a 2D vector image
    ## otherwise it will be treaded as a 3D image
    image = sitk.GetImageFromArray(array, isVector=False)

    if 'direction' in param: image.SetDirection(param['direction'])
    if 'spacing' in param: image.SetSpacing(param['spacing'])
    if 'origin' in param: image.SetOrigin(param['origin'])

    if 'type' not in param:
        sitk.WriteImage(image, target_path, True)
    else:
        ## 如果是标签，按照MITK要求改为unsigned_char/unsigned_short型 [sitk.sitkUInt8]
        sitk.WriteImage(sitk.Cast(image, param['type']), target_path, True)


## 将numpy数组保存为普通的2D图像，支持.jpg, .jpeg, .tif
def save_normal_image(array, target_path):
    '''
    将得到的数组保存为普通的2D图像
    :param array: 想要保存的图像数组
    :param target_path: 保存的文件路径，注意：一定要带后缀，E.g.,.jpg,.png,.tif
    :return: None 无返回值
    '''
    image = Image.fromarray(array)
    image.save(target_path)


def save_csv(file, rows, headers=None, delimiter=','):
    import csv
    with open(file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        if headers is not None:
            writer.writerow(headers)
        writer.writerows(rows)


## save dict as json
def save_json(dicts, file, indent=2):
    import json
    info = json.dumps(dicts, indent=indent, ensure_ascii=False)
    with open(file, 'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
        f.write(info)


## save obj as pickle
def save_pickle(obj, file):
    import pickle
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


## save points clouds as mps (mitk only)
def save_mps(points, target_path, offset=(0, 0, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1)):
    points = np.asarray(points)
    direction = """
     <IndexToWorld type="Matrix3x3" m_0_0="{}" m_0_1="{}" m_0_2="{}" m_1_0="{}" m_1_1="{}" m_1_2="{}" m_2_0="{}" m_2_1="{}" m_2_2="{}"/>
     """.format(direction[0], direction[1], direction[2],
                direction[3], direction[4], direction[5],
                direction[6], direction[7], direction[8])

    offset = """<Offset type="Vector3D" x="{}" y="{}" z="{}"/>
    """.format(offset[0], offset[1], offset[2])

    minx, miny, minz = points[:, 0].min(), points[:, 1].min(), points[:, 2].min()
    maxx, maxy, maxz = points[:, 0].max(), points[:, 0].max(), points[:, 0].max()
    bounds = """
    <Bounds>
        <Min type="Vector3D" x="{}" y="{}" z="{}"/>
        <Max type="Vector3D" x="{}" y="{}" z="{}"/>
    </Bounds>
    """.format(minx, miny, minz, maxx, maxy, maxz)

    strpoints = ""
    for index, point in enumerate(points):
        x, y, z = point
        strpoints += """
        <point>
            <id>{}</id>
            <specification>0</specification>
            <x>{}</x>
            <y>{}</y>
            <z>{}</z>
        </point>
        """.format(index, x, y, z)

    mps = """
    <?xml version="1.0" encoding="UTF-8"?>
    <point_set_file>
    <file_version>0.1</file_version>
    <point_set>
        <time_series>
            <time_series_id>0</time_series_id>
            <Geometry3D ImageGeometry="false" FrameOfReferenceID="0">
                {}
                {}
                {}
            </Geometry3D>
            {}
        </time_series>
    </point_set>
    </point_set_file>
    """.format(direction, offset, bounds, strpoints)

    with open(target_path, 'w', encoding='utf-8')as file:
        file.write(mps)


def example_get_dicom_image():
    array, o, s, d, t = get_dicom_image(dire="D:/Projects/Pancreas/sort/SRS00013")
    save_medical_image(array, o, s, d, './test.nrrd', t)
    # import numpy as np
    # print(np.asarray(array).shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(array[0], cmap=plt.cm.bone)
    # plt.show()


def example_get_dicom_images():
    root = './unzip'
    dires = get_files_name(dire=root)

    for dire in dires:
        num, images, proper = get_dicom_images(dire=os.path.join(root, dire))

        for index in range(num):
            image, param = get_medical_image(images[index])

            if 'CT' in proper[index]['0008|103e']:
                save_medical_image(image, target_path=os.path.join('./data/CT', "{}.nrrd".format(dire)), param=param)

            if 'T1' in proper[index]['0008|103e']:
                save_medical_image(image, target_path=os.path.join('./data/T1', "{}.nrrd".format(dire)), param=param)

            if 'T2' in proper[index]['0008|103e']:
                save_medical_image(image, target_path=os.path.join('./data/T2', "{}.nrrd".format(dire)), param=param)


def example_get_medical_image():
    array, _, _, _, _ = get_medical_image("E:/Work/Database/Liver/LiTS/segmentation-0.nii")
    import numpy as np
    print(np.asarray(array).shape)
    import matplotlib.pyplot as plt
    plt.imshow(array[50], cmap=plt.cm.bone)
    plt.show()


def example_get_normal_image():
    array = get_normal_image('E:/Work/Github/Reference_Model/15-Unet/data/train/images/0.tif')
    print(type(array))
    import numpy as np
    print(np.asarray(array).shape)
    import matplotlib.pyplot as plt
    plt.imshow(array, cmap=plt.cm.bone)
    plt.show()


def example_save_medical_image():
    array, origin, spacing, direction, image_type = get_medical_image(
        "E:/Work/Database/Liver/LiTS/segmentation-0.nii")
    save_medical_image(array=np.asarray(array), origin=origin, spacing=spacing, direction=direction,
                       target_path='./result.nrrd', type=image_type)


def example_save_normal_image():
    array = get_normal_image('E:/Work/Github/Reference_Model/15-Unet/data/train/images/5.tif')
    save_normal_image(array, './r.tif')


def example_get_files_name():
    dire = "E:/Work/Database/Liver/ISICDM2019/train/"
    files = get_files_name(dire)
    pathes = []
    for file in files:
        pathes.append(os.path.join(dire, file))
    print(pathes)


############################################################################################
##                  整理原始DICOM文件夹，获取DICOM信息
############################################################################################


## 搜索DICOMDIR文件
def search_dicomdir(dire):
    '''
    获取一个文件夹下所有DICOMDIR文件的目录
    :param dire: 总文件夹，找寻该文件夹下所有DICOMDIR的位置
    :return: dicomdirpaths，DICOMDIR所在的位置
    '''

    dicomdirpaths = []

    def SearchDICOMDIR(folder):
        files = get_files_name(folder)
        if 'DICOMDIR' in files:
            dicomdirpaths.append(os.path.normpath(os.path.join(folder, 'DICOMDIR')))
        else:
            for file in files:
                tmp_dire = os.path.join(folder, file)
                if os.path.isdir(tmp_dire):
                    SearchDICOMDIR(tmp_dire)

    SearchDICOMDIR(dire)

    return dicomdirpaths


## 读DICOMDIR文件
def get_dicomdir_info(path):
    '''
    读取DICOMDIR中的信息，主要包含以下信息：
    PatientID, PatientName, StudyDate, StudyDescription, StudyID, SerialNumber. SeriesCount,   SeriesDescription, SeriesFolder, SeriesModality
    病人ID，   病人名字，    病历日期，  病历描述，        病历ID，  序列号，      序列包含图片数， 序列描述，        序列所在文件夹，序列模态
    但因为DICOMDIR信息不全，所以这里只读取关键的PatientID，PatientName，SeriesFolder信息
    :param path: DICOMDIR文件的路径
    :return: list,每一个序列的以上基本信息
    DICOMDIR文件格式说明：https://www.medicalconnections.co.uk/kb/DICOMDIR/
    https://pydicom.github.io/pydicom/dev/auto_examples/input_output/plot_read_dicom_directory.html
    四级结构：PATIENT –> STUDY –> SERIES –> IMAGE
    这里只考虑一个序列的文件全在同一个文件夹的情况
    '''
    assert os.path.split(path)[1] == 'DICOMDIR', 'file is not DICOMDIR'

    from pydicom.filereader import read_dicomdir
    from os.path import dirname, join, split

    ## 读取DICOMDIR文件
    reader = read_dicomdir(path)

    ## DICOM文件所在的文件夹
    dire = dirname(path)

    ## 得到结果：PatientID, PatientName, SeriesFolder
    names = []
    ids = []
    folders = []

    ## 读取PATIENT的记录
    for record in reader.patient_records:

        ## 查找病人的 PatientID 与 PatientName
        assert hasattr(record, 'PatientID') and \
               hasattr(record, 'PatientName'), "record doesn't have patient ID or patient Name"
        id = "{}".format(record.PatientID)
        if type(record.PatientName).__name__ == 'PersonName3':
            name = "{}".format(record.PatientName)
            if name.find('=') != -1:
                name = name[:name.find('=')].upper()
        elif type(record.PatientName).__name__ == 'MultiValue':
            name = "{}".format(record.PatientName[0])
            if name.find('=') != -1:
                name = name[:name.find('=')].upper()
        else:
            assert type(record.PatientName).__name__ == 'PersonName3' and \
                   type(record.PatientName).__name__ == 'MultiValue', \
                "Cannot find correct type of record.PatientName"

        names.append(name)
        ids.append(id)
        folder = []

        ## 读取STUDY的记录
        studies = record.children
        for study in studies:
            ## 读取SERIES记录
            all_series = study.children
            for series in all_series:
                ## 读取文件所在的文件夹，这里认为同一个序列里的文件都在同一个文件夹中，所以只去文件夹路径，不取单个文件的路径
                ## 如果发现同一个序列的图片在不同的文件夹中会报警
                image_records = series.children
                image_filenames = [split(join(os.path.normpath(dire),
                                              *image_rec.ReferencedFileID))[0] for image_rec in image_records]
                image_folder = np.unique(image_filenames)
                assert image_folder.shape[0] == 1, \
                    'dicom image does not storage in only one folder DICOM Path: {}'.format(path)
                image_folder = os.path.normpath(image_folder[0])
                folder.append(image_folder)
        folders.append(folder)

    return names, ids, folders


## 得到一组dicom序列信息
def get_dicom_info(dire):
    '''
    加载一组dicom序列图像
    :param dire: dicom序列所在的文件夹路径，E.g. "E:/Work/Database/Teeth/origin/1/"
    :return: (array,origin,spacing,direction)
    '''

    assert os.path.exists(dire), "{} is not existed".format(dire)
    assert os.path.isdir(dire), "{} is not a directory".format(dire)

    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dire)
    if not series_IDs:
        print("ERROR: given directory \"" + dire + "\" does not contain a DICOM series.")
        sys.exit(1)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dire, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()

    Tags = [
        ##########################  Patient Tag  ##########################
        "0010|0010",  # Patient Name        患者姓名                    (PN)
        "0010|0020",  # Patient ID          患者ID                      (LO)
        "0010|0030",  # Patient Birth Date  患者出生日期                (DA)
        "0010|0032",  # Patient Birth Time  患者出生时间                (TM)
        "0010|0040",  # Patient Sex         患者性别                    (CS)
        "0010|1010",  # Patient age         患者年龄（拍照时）           (AS)
        "0010|1030",  # Patient Weight      患者体重                    (DS)
        "0010|21c0",  # Pregnancy Status    怀孕状态                    (US)
        "0008|0070",  # Manufacturer        医院选用的设备              （LO）
        ###########################  Study Tag  ###########################
        "0008|0050",  # Accession Number    检查号（RIS生成序号）        (SH)
        "0020|0010",  # Study ID            检查ID                      (SH)
        "0020|000d",  # Study Instance UID  检查实例号（标记唯一不同检查）(UI)
        "0008|0020",  # Study Date          检查开始日期                 (DA)
        "0008|0030",  # Study Time          检查开始时间                 (TM)
        "0008|0061",  # Modalities in Study 检查模态                     (CS)
        "0008|0015",  # Body Part Examined  检查部位                     (CS)
        "0008|1030",  # Study Description   检查描述                     (LO)
        ###########################  Series Tag  ###########################
        "0020|0011",  # Series Number       序列号                       (IS)
        "0020|000e",  # Series Instance UID 序列实例号                   (UI)
        "0020|4000",  # Image Comments      图像标注（长海会在这里储存时态：静脉期/动脉期/延迟期）
        "0008|0060",  # Modality            检查模态                     (CS)
        "0008|103e",  # Series Description  检查描述和说明               (LO)
        "0008|0021",  # Series Date         检查日期                     (DA)
        "0008|0031",  # Series Time         检查时间                     (TM)
        "0008|0032",  # Acquisition Time    探测时间 (截获时间)
        "0020|0032",  # Image Position      图像位置(图像左上角在空间中的位置:mm) (DS)
        "0018|0050",  # Slice Thickness     层厚                         (DS)
        "0018|0088",  # Spacing between Slices 层间距:mm                 (DS)
        "0020|1041",  # Slice location      实际的相对位置                (DS)
        "0018|0023",  # MR Acquisition                                   (CS)
        "0018|0015",  # Body Part Examined  身体部位                      (CS)
        ###########################  Image Tag  ###########################
        "0008|0008",  # Image Type          图像类型                      (CS)
        "0008|0018",  # SOP Instance UID    SOP实例的UID                  (CS)
        "0008|0023",  # Content Date        影像拍摄的日期                (DA)
        "0008|0033",  # Content Time        影像拍摄的时间                (TM)
        "0020|0013",  # Image Number        图像码                       (IS)
        "0028|0002",  # Sample Per Pixel    图像采样率                   (US)
        "0028|0004",  # Photometric Interpretation 光度计解释（判断是否为灰度）(CS)
        "0028|0010",  # Rows                图像的总行数                  (US)
        "0028|0011",  # Columns             图像的总列数                  (US)
        "0028|0030",  # Pixel Spacing       像素间距                      (DS)
        "0028|0100",  # Bits Allocated      存储一个像素时分配的位数       (US)
        "0028|0101",  # Bits Stored         存储一个像素时的位数           (US)
        "0028|0102",  # High Bit            高位                          (US)
        "0028|0103",  # Pixel Representation像素的表现类型                 (US)
        "0028|1050",  # Windows Center      窗位                          (DS)
        "0028|1051",  # Windows Width       窗宽                          (DS)
        "0028|1052",  # Rescale Intercept   截距 (b)                      (DS)
        "0028|1053",  # Rescale Slope       斜率 (m)                      (DS)
        "0028|1054",  # Rescale Type        斜率                          (DS)
        # Output units = m*SV + b. #
    ]

    Description = [
        ##########################  Patient Tag  ##########################
        "Patient Name",
        "Patient ID",
        "Patient Birth Date",
        "Patient Birth Time",
        "Patient Sex",
        "Patient age",
        "Patient Weight",
        "Pregnancy Status",
        "Manufacturer",
        ###########################  Study Tag  ###########################
        "Accession Number",
        "Study ID",
        "Study Instance UID",
        "Study Date",
        "Study Time",
        "Modalities in Study",
        "Body Part Examined",
        "Study Description",
        ###########################  Series Tag  ###########################
        "Series Number",
        "Series Instance UID",
        "Image Comments",
        "Modality",
        "Series Description",
        "Series Date",
        "Series Time",
        "Acquisition Time",
        "Image Position",
        "Slice Thickness",
        "Spacing between Slice",
        "Slice location",
        "MR Acquisition",
        "Body Part Examined",
        ###########################  Image Tag  ###########################
        "Image Type",
        "SOP Instance UID",
        "Content Date",
        "Content Time",
        "Image Number",
        "Sample Per Pixel",
        "Photometric Interpretation",
        "Rows",
        "Columns",
        "Pixel Spacing",
        "Bits Allocated",
        "Bits Stored",
        "High Bit",
        "Pixel Representation",
        "Windows Center",
        "Windows Width",
        "Rescale Intercept",
        "Rescale Slope",
        "Rescale Type",
    ]

    info = {}

    for index, k in enumerate(Tags):
        if series_reader.HasMetaDataKey(0, k):
            str = series_reader.GetMetaData(0, k)
            ## 处理中文字符的乱编码
            if index == 0 and str.find('=') != -1:
                str = str[:str.find('=')].upper()
            info[Description[index]] = str
            # print("{} {:30s} -- {}#".format(k, Description[index].replace(" ",""), series_reader.GetMetaData(0, k)))
        # else:
        #     info[Description[index]] = None

    return info


## 整理包含乱序文件的DICOMDIR
def sort_disorder_dicomdir(path, out_dire):
    '''
    前提：这个文件夹中的DICOMDIR中一个序列中的文件散落在不同的文件夹中
    以拷贝的方式在目标目录中按照 NAME-ID → Series 的数据结构重建文件夹，并将ID,Name,SeriesFolder的信息统计返回
    :param path: 包含乱序DICOMDIR的文件夹
    :param out_dire: 目标文件夹
    :return: ID, Name, SeriesFolder信息
    '''
    import shutil

    assert os.path.split(path)[1] == 'DICOMDIR', 'file is not DICOMDIR'

    from pydicom.filereader import read_dicomdir

    ## 读取DICOMDIR文件
    reader = read_dicomdir(path)

    ## DICOM文件所在的文件夹
    dire = os.path.normpath(os.path.dirname(path))

    ## 得到结果：PatientID, PatientName, SeriesFolder
    names = []
    ids = []
    folders = []

    ## 读取PATIENT的记录
    for record in reader.patient_records:

        ## 查找病人的 PatientID 与 PatientName
        assert hasattr(record, 'PatientID') and \
               hasattr(record, 'PatientName'), "record doesn't have patient ID or patient Name"
        # id = "{}".format(record.PatientID)
        # name = "{}".format(record.PatientName)
        # name = name[:name.find('=')].upper()

        id = "{}".format(record.PatientID)
        if type(record.PatientName).__name__ == 'PersonName3':
            name = "{}".format(record.PatientName)
            if name.find('=') != -1:
                name = name[:name.find('=')].upper()
        elif type(record.PatientName).__name__ == 'MultiValue':
            name = "{}".format(record.PatientName[0])
            if name.find('=') != -1:
                name = name[:name.find('=')].upper()
        else:
            assert type(record.PatientName).__name__ == 'PersonName3' and \
                   type(record.PatientName).__name__ == 'MultiValue', \
                "Cannot find correct type of record.PatientName"

        names.append(name)
        ids.append(id)
        folder = []

        ## 在相应文件夹下创建该序列的文件夹目录
        patient_dire = os.path.join(out_dire, "{}-{}".format(name, id))
        os.makedirs(patient_dire, exist_ok=True)

        ## 读取STUDY的记录
        studies = record.children
        for study in studies:
            ## 读取SERIES记录
            all_series = study.children
            for series in all_series:
                ## 该文件夹下的序列数
                num = len(os.listdir(patient_dire))
                series_dire = os.path.normpath(os.path.join(patient_dire, "SRS000{}".format('{}'.format(num).zfill(2))))
                ## 这里不允许该文件夹被提前建立
                os.makedirs(series_dire)

                ## 读取一个序列的所有图像的路径
                image_records = series.children
                image_filenames = [os.path.join(dire, *image_rec.ReferencedFileID)
                                   for image_rec in image_records]

                for index, path in enumerate(image_filenames):
                    shutil.copyfile(src=path,
                                    dst=os.path.join(series_dire, '{}.dcm'.format('{}'.format(index).zfill(3))))

                folder.append(series_dire)
        folders.append(folder)

    return names, ids, folders


## 汇总DICOMDIR文件信息
def sum_dicomdir_info(dire, json_dire, disorder_out_dire):
    '''
    将一个文件夹中的DICOMDIR信息读取、汇总起来
    一个序列的文件必须集中在同一个文件夹中，否则会报错退出
    :param dire: 文件夹路径
    :param json_dire: 为防止中间出现的意外错误导致暂停，每完成一个DICOMDIR的扫描就建立一个json信息文件
    :param disorder_out_dire: 乱序DICOMDIR输出的整理文件夹
    :return: None
    '''
    from tqdm import tqdm
    import warnings
    import pydicom
    import json

    ## 创建json信息文件夹
    os.makedirs(json_dire, exist_ok=True)

    ## 创建无序DICOMDIR整理文件夹
    os.makedirs(disorder_out_dire, exist_ok=True)

    ## 因为有中文名字无法编码的warning，所以忽略warning
    warnings.filterwarnings("ignore")

    ## 遍历每一个文件夹
    print(" Search DICOMDIR ...")
    paths = search_dicomdir(dire)

    ## 遍历每一个DICOMDIR文件
    print(" Traverse DICOMDIR ...")
    for index, path in tqdm(enumerate(paths)):
        ## 一个文件夹中所有DICOMDIR的信息
        info = {}

        ## 获取DICOMDIR中的序列信息
        ## name, id, folders = get_dicomdir_info(path)
        ## 如果DICOMDIR中记录的同一个序列的所有文件均在同一个文件夹中，
        ## 获取该序列的病人的name,id,以及该序列的路径即可

        ## name, id, folders = sort_disorder_dicomdir(path, out_dire=disorder_out_dire)
        ## 如果DICOMDIR中记录的同一个序列的所有文件分布在不同的文件中，
        ## 那么在disorder_out_dire文件中，将同一个序列的文件以拷贝的形式归整到同一个文件夹
        ## 获取该序列病人的name,id，以及新的归整序列文件夹的路径
        try:
            names, ids, folders = get_dicomdir_info(path)
        except pydicom.errors.InvalidDicomError:
            ## 可能存在DICOMDIR文件读取失败，跳过
            print("DICOM error, path: {}".format(path))
            continue
        except AssertionError as assert_content:
            print("{}".format(assert_content))
            if "{}".format(assert_content).find("dicom image does not storage in only one folder") != -1:
                print('Find Disorder DICOMDIR, Do sorting the disorder dicom ...')
                try:
                    names, ids, folders = sort_disorder_dicomdir(path, out_dire=disorder_out_dire)
                except Exception as e:
                    ## 可能存在拷贝中遗漏某些文件，或者其他奇奇怪怪的错误，先跳过，后面再进行处理
                    print(
                        "\n\n**************************** sort disorder dicomdir error *********************************")
                    print("Error info: {}".format(e))
                    print("Error index: {}".format(index))
                    continue

        ## 将获取的dicomdir中的关键信息保存下来
        for index_id, id in enumerate(ids):
            name = names[index_id]
            folder = folders[index_id]
            if id not in info:
                info[id] = {'name': name, "folders": folder}
            else:
                try:
                    assert name == info[id]["name"], \
                        "\nThe same id maps different patient name! name 1:{} name 2:{}".format(name, info[id]["name"])
                except AssertionError:
                    print(
                        "\nThe same id maps different patient name! name 1:{} name 2:{}".format(name, info[id]["name"]))
                    print("DICOM Path: {}".format(path))
                    pass
                    # print('Has DICOMDIR file with the same id (name) id: {} name: {}'.format(id, name))
                    # info[id]["folders"] += folders

                print('Has DICOMDIR file with the same id (name) id: {} name: {}'.format(id, name))
                info[id]["folders"] += folder

        info = json.dumps(info, indent=2, ensure_ascii=False)
        with open(os.path.join(json_dire, "{}.json".format(index)), 'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
            f.write(info)


## 汇总DICOM信息
def sum_dicomjson_info(path, tmp_dire):
    '''
    汇总DICOM信息，DICOM序列的路径保存在json文件中
    :param path: .json文件的路径
    :param tmp_dire: 暂存.json所存储在的文件加
    为了防止DICOM文件众多而出现的意外错误，每一个病人建立一个json，过后使用集合将json合并
    :return:
    '''

    assert os.path.exists(path), "{} doesn't exists"
    os.makedirs(tmp_dire, exist_ok=True)

    ## 读取在文件中的各个序列的文件夹信息
    from tqdm import tqdm
    import json

    with open(path, 'r', encoding='utf-8') as f:
        dicomdir_info = json.load(f)

    ## 防止过热，每50个休息30s
    pid_num = 0
    ## 读取每一个病人的信息
    for index, pid in enumerate(tqdm(natsort.natsorted(dicomdir_info))):
        pinfo = dicomdir_info[pid]

        ## 读取每一个序列的信息
        series_info = {}
        pseries = pinfo["folders"]
        for dire in pseries:
            try:
                sinfo = get_dicom_info(dire=dire)
            except:
                continue

            assert dire not in sinfo, "same series of one patient, dire:{}".format(dire)
            series_info[dire] = sinfo

        ## 病人的信息
        patient_info = {}
        patient_info["name"] = pinfo["name"]
        patient_info["series"] = series_info
        dicom_info = {}
        dicom_info[pid] = patient_info

        ## 将病人的信息存储下来
        info = json.dumps(dicom_info, indent=2, ensure_ascii=False)
        with open(os.path.normpath(os.path.join(tmp_dire, "{}.json".format(index))),
                  'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
            f.write(info)

        # ## 每20个休息30s
        # pid_num += 1
        # if pid_num == 50:
        #     print("Sleep time 30s...")
        #     pid_num = 0
        #     time.sleep(30)


## 将一个文件夹中的.json文件汇总为一个.json文件
def sum_json(dire, json_file):
    from tqdm import tqdm
    def Merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res

    import json
    info = {}
    files = get_files_name(dire=dire)
    for file in tqdm(files):
        path = os.path.join(dire, file)
        with open(path, 'r', encoding='utf-8') as f:
            tmp = json.load(f)
        info = Merge(info, tmp)

    info = json.dumps(info, indent=2, ensure_ascii=False)
    with open(json_file, 'w', encoding='utf-8') as f:  # 使用.dumps()方法时，要写入
        f.write(info)


def example_sum_dicomdir_info():
    names = [
        "2013DWI",
        "2019-01-23PDAC",
        "20190124jiangyeliu",
        "DICOM",
        "nianyeliu",
        "PC",
        "PC2015",
        "Pancreatic Tumor",
        "1PDAC",
        "NORMAL",
    ]

    for name in names:
        print('Name: {}'.format(name))
        sum_dicomdir_info("H:/RawData/" + name,
                          "./json/{}.json".format(name),
                          disorder_out_dire="H:/SortData/")


def example_sum_dicomjson_info():
    import time
    names = [
        # "2013DWI",
        # "2019-01-23PDAC",
        # "20190124jiangyeliu",
        # "DICOM",
        # "nianyeliu",
        # "PC",
        # "PC2015",
        # "Pancreatic Tumor",
        ## "1PDAC",
        "NORMAL",
    ]

    for name in names:
        print('####################################################################')
        print(name)
        print('####################################################################')
        sum_dicomjson_info("./json/series/{}.json".format(name), tmp_dire='./json/{}'.format(name))
        time.sleep(120)


if __name__ == '__main__':
    example_get_dicom_image()
    # example_get_medical_image()
    # example_get_normal_image()
    # example_save_medical_image()
    # example_save_normal_image()
    # example_get_files_name()
    # example_sum_dicomdir_info()
