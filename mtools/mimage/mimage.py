import numpy as np


## 归一化 (0,1)标准化
def norm_zero_one(array, span=None):
    '''
    根据所给数组的最大值、最小值，将数组归一化到0-1
    :param array: 数组
    :return: array: numpy格式数组
    '''
    array = np.asarray(array).astype(np.float32)
    if span is None:
        mini = array.min()
        maxi = array.max()
    else:
        mini = span[0]
        maxi = span[1]
        array[array < mini] = mini
        array[array > maxi] = maxi

    range = maxi - mini

    def norm(x):
        return (x - mini) / range

    return np.asarray(list(map(norm, array))).astype(np.float32)


## 归一化，Z-score标准化
def norm_z_score(array):
    '''
    根据所给数组的均值和标准差进行归一化，归一化结果符合正态分布，即均值为0，标准差为1
    :param array: 数组
    :return: array: numpy格式数组
    '''
    array = np.asarray(array).astype(np.float)
    mu = np.average(array)  ## 均值
    sigma = np.std(array)  ## 标准差

    def norm(x):
        return (x - mu) / sigma

    return np.asarray(list(map(norm, array))).astype(np.float), mu, sigma
