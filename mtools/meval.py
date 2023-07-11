#!/usr/bin/python3
# -*- coding:utf-8 -*-

###################################################################
## Author: MiaoMiaoYang
## Created: 20.03.20
## Last Changed: 20.03.20
###################################################################

###################################################################
## 医学图像评价方法
## Cite:
## @article{taha2015metrics,
##   title={Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool},
##   author={Taha, Abdel Aziz and Hanbury, Allan},
##   journal={BMC medical imaging},
##   volume={15},
##   number={1},
##   pages={29},
##   year={2015},
##   publisher={BioMed Central}
## }
###################################################################

import math
import numpy as np


## dice
def get_dice(y_pred, y_true):
    '''
    计算两个二值数组之间的dice系数
    math::
    $$ DICE=\frac{2TP}{2TP+FP+FN}=\frac{2|A\cap B|}{|A|+|B|} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: dice [0,1] 0-完全无重叠部分 1-完全重叠在一起
    '''
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    intersection = np.count_nonzero(y_pred & y_true)

    size_i1 = np.count_nonzero(y_pred)
    size_i2 = np.count_nonzero(y_true)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


## jaccard
def get_jac(y_pred, y_true):
    '''
    计算两个二值数组之间的jaccard系数
    math::
    $$ Jaccard=\frac{|A\cap B|}{|A\cup B|}=\frac{DICE}{2-DICE} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: jaccard [0,1] 0-完全无重叠部分 1-完全重叠在一起
    '''
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    intersection = np.count_nonzero(y_pred & y_true)
    union = np.count_nonzero(y_pred | y_true)

    try:
        jaccard = float(intersection) / float(union)
    except ZeroDivisionError:
        jaccard = 0.0

    return jaccard


## true positive rate
def get_tpr(y_pred, y_true):
    '''
    真阳率：将正类预测为正类的比例
    True positive rate
    math::
    $$ TPR = \frac{TP}{TP+FN} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: true positive rate
    '''
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = np.count_nonzero(y_pred & y_true)
    fn = np.count_nonzero(~y_pred & y_true)

    true_positive_rate = float(tp) / (float(tp) + float(fn))

    return true_positive_rate


## true negative rate
def get_tnr(y_pred, y_true):
    '''
    真阴率：将负类预测为负类的比例
    True positive rate
    math::
    $$ TNR = \frac{TN}{TN+FP} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: true negative rate
    '''
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tn = np.count_nonzero(~y_pred & ~y_true)
    fp = np.count_nonzero(y_pred & ~y_true)

    true_negative_rate = float(tn) / (float(tn) + float(fp))
    return true_negative_rate


## false positive rate
def get_fpr(y_pred, y_true):
    return 1. - get_tnr(y_pred, y_true)


## false negative rate
def get_fnr(y_pred, y_true):
    return 1. - get_tpr(y_pred, y_true)


## recall
def get_recall(y_pred, y_true):
    '''
    召回率: 覆盖面的度量 - 度量多少个正例被正确分类
    recall
    math::
    $$ Recall = \frac{TP}{TP+FN} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: true positive rate
    '''
    return get_tpr(y_pred, y_true)


def get_multi_recall(y_pred, y_true, labels=None):
    from sklearn.metrics import recall_score
    return recall_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


## sensitivity
def get_sensitivity(y_pred, y_true):
    '''
    敏感度: 衡量了分类器对正类的识别能力
    recall
    math::
    $$ Recall = \frac{TP}{TP+FN} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: true positive rate
    '''
    return get_tpr(y_pred, y_true)


## specificity
def get_specificity(y_pred, y_true):
    '''
    特异性: 衡量分类器对负类的识别能力
    specificity
    math::
    $$ specificity = \frac{TN}{TN+FP} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: true negative rate
    '''
    return get_tnr(y_pred, y_true)


## precision
def get_precision(y_pred, y_true):
    '''
    精确率，精度：衡量被分类为正例中真实为正例的比例
    precision
    math::
    $$ Precision = \frac{TP}{TP+FP} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: precision
    '''
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = np.count_nonzero(y_pred & y_true)
    fp = np.count_nonzero(y_pred & ~y_true)

    precision = float(tp) / (float(tp) + float(fp))

    return precision


def get_multi_precision(y_pred, y_true, labels=None):
    from sklearn.metrics import precision_score
    return precision_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


def get_multi_accuracy(y_pred, y_true):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true=y_true, y_pred=y_pred)


def get_each_accuracy(y_pred, y_true):
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_true, y_pred)
    return matrix.diagonal() / matrix.sum(axis=1)


## accuracy
def get_accuracy(y_pred, y_true):
    '''
    准确率：分对的数目/所有数目的比
    accuracy
    math::
    $$ Accuracy = \frac{TP+TN}{TP+TN+FP+FN} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: accuracy
    '''
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = np.count_nonzero(y_pred & y_true)
    tn = np.count_nonzero(~y_pred & ~y_true)
    fp = np.count_nonzero(y_pred & ~y_true)
    fn = np.count_nonzero(~y_pred & y_true)

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(tn) + float(fp) + float(fn))
    return accuracy


## f1
def get_multi_f1(y_pred, y_true, labels=None):
    from sklearn.metrics import f1_score
    return f1_score(y_true=y_true, y_pred=y_pred, average='macro', labels=labels)


## fall out
def get_fallout(y_pred, y_true):
    return 1.0 - get_tnr(y_pred, y_true)


## global consistency error
def get_gce(y_pred, y_true):
    '''
    Global Consistency Error
    $R(S,x)$ is defined as the set of all voxels that
    reside in the same region of segmentation $S$
    where the voxel $x$ resides. [Taha,2015]
    $E(S_{t},S_{g},x)=\frac{|R(S_{T},x)/R(S_{g},x)|}{R(S_{T},x)}$
    $$ GCE(S_{t},S_{g})=\frac{1}{n}\min\{\sum \limits_{i}^{n}(S_{t},S_{g},x_{i}),\sum \limits_{i}^{n}(S_{g},S_{t},x_{i})\} $$
    $$ GCE(S_{t},S_{g})=\frac{1}{n}\min\{\frac{FN(FN+2TP)}{TP+FN}+\frac{FP(FP+2TN)}{TN+FP},\frac{FP(FP+2TP)}{TP+FP}+\frac{FN(FN+2TN)}{TN+FN}\} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: global consistency error
    '''
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = float(np.count_nonzero(y_pred & y_true))
    tn = float(np.count_nonzero(~y_pred & ~y_true))
    fp = float(np.count_nonzero(y_pred & ~y_true))
    fn = float(np.count_nonzero(~y_pred & y_true))

    n = tp + tn + fp + fn

    e1 = (fn * (fn + 2 * tp) / (tp + fn) + fp * (fp + 2 * tn) / (tn + fp)) / n
    e2 = (fp * (fp + 2 * tp) / (tp + fp) + fn * (fn + 2 * tn) / (tn + fn)) / n

    return min(e1, e2)


## volumetric similarity
def get_vs(y_pred, y_true):
    '''
    Volumetric Similarity
    math:
    $$ VS = 1-\frac{||A|-|B||}{|A|+|B|}=1-\frac{|FN-FP|}{2TP+FP+FN} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: global consistency error
    '''

    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = np.count_nonzero(y_pred & y_true)
    fn = np.count_nonzero(~y_pred & y_true)
    fp = np.count_nonzero(y_pred & ~y_true)

    try:
        vs = 1 - (abs(fn - fp) / float(2 * tp + fp + fn))
    except ZeroDivisionError:
        vs = 0.0

    return vs


## Rand Index
def get_ri(y_pred, y_true):
    '''
    Rand Index
    math:
    $$
    \begin{cases}
        & a = \frac{1}{2}[TP(TP-1)+FP(FP-1)+TN(TN-1)+FN(FN-1)] \\
        & b = \frac{1}{2}[(TP+FN)^{2}+(TN+FP)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & c = \frac{1}{2}[(TP+FP)^{2}+(TN+FN)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & d = n(n-1)/2-(a+b+c)
    \end{cases}
    $$
    $$ RI(A,B) = \frac{a+b}{a+b+c+d} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: rand index
    '''

    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = float(np.count_nonzero(y_pred & y_true))
    tn = float(np.count_nonzero(~y_pred & ~y_true))
    fp = float(np.count_nonzero(y_pred & ~y_true))
    fn = float(np.count_nonzero(~y_pred & y_true))

    n = tp + tn + fp + fn

    a = (tp * (tp - 1) + fp * (fp - 1) + tn * (tn - 1) + fn * (fn - 1)) / 2
    b = ((tp + fn) ** 2 + (tn + fp) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    c = ((tp + fp) ** 2 + (tn + fn) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    d = n * (n - 1) / 2 - (a + b + c)

    ri = (a + b) / (a + b + c + d)

    return ri


## Adjusted Rand Index
def get_ari(y_pred, y_true):
    '''
    Adjusted Rand Index
    math:
    $$
    \begin{cases}
        & a = \frac{1}{2}[TP(TP-1)+FP(FP-1)+TN(TN-1)+FN(FN-1)] \\
        & b = \frac{1}{2}[(TP+FN)^{2}+(TN+FP)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & c = \frac{1}{2}[(TP+FP)^{2}+(TN+FN)^{2}-(TP^{2}+TN^{2}+FP^{2}+FN^{2})] \\
        & d = n(n-1)/2-(a+b+c)
    \end{cases}
    $$
    $$ ARI(A,B) = \frac{2(ad-bc)}{c^{2}+b^{2}+2ad+(a+d)(c+b)} $$
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: adjusted rand index
    '''

    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = float(np.count_nonzero(y_pred & y_true))
    tn = float(np.count_nonzero(~y_pred & ~y_true))
    fp = float(np.count_nonzero(y_pred & ~y_true))
    fn = float(np.count_nonzero(~y_pred & y_true))

    n = tp + tn + fp + fn

    a = (tp * (tp - 1) + fp * (fp - 1) + tn * (tn - 1) + fn * (fn - 1)) / 2
    b = ((tp + fn) ** 2 + (tn + fp) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    c = ((tp + fp) ** 2 + (tn + fn) ** 2 - (tp ** 2 + tn ** 2 + fp ** 2 + fn ** 2)) / 2
    d = n * (n - 1) / 2 - (a + b + c)

    ari = (2 * (a * d - b * c)) / (c ** 2 + b ** 2 + 2 * a * d + (a + d) * (c + b))
    return ari


## Mutual Information
def get_mi(y_pred, y_true):
    '''
    Mutual Information
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: adjusted rand index
    '''

    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = float(np.count_nonzero(y_pred & y_true))
    tn = float(np.count_nonzero(~y_pred & ~y_true))
    fp = float(np.count_nonzero(y_pred & ~y_true))
    fn = float(np.count_nonzero(~y_pred & y_true))

    n = tp + tn + fp + fn

    row1 = tn + fn
    row2 = fp + tp
    H1 = - ((row1 / n) * math.log(row1 / n, 2) + (row2 / n) * math.log(row2 / n, 2))

    col1 = tn + fp
    col2 = fn + tp
    H2 = - ((col1 / n) * math.log(col1 / n, 2) + (col2 / n) * math.log(col2 / n, 2))

    p00 = 1 if tn == 0 else tn / n
    p01 = 1 if fn == 0 else fn / n
    p10 = 1 if fp == 0 else fp / n
    p11 = 1 if tp == 0 else tp / n
    H12 = - ((tn / n) * math.log(p00, 2) +
             (fn / n) * math.log(p01, 2) +
             (fp / n) * math.log(p10, 2) +
             (tp / n) * math.log(p11, 2))
    MI = H1 + H2 - H12
    return MI


## Variation of Information
def get_voi(y_pred, y_true):
    '''
    Variation Information
    :param y_pred: 可以二值化的数组，背景为0，前景为非0
    :param y_true: 可以二值化的数组，背景为0，前景为非0
    :return: adjusted rand index
    '''

    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = float(np.count_nonzero(y_pred & y_true))
    tn = float(np.count_nonzero(~y_pred & ~y_true))
    fp = float(np.count_nonzero(y_pred & ~y_true))
    fn = float(np.count_nonzero(~y_pred & y_true))

    n = tp + tn + fp + fn

    row1 = tn + fn
    row2 = fp + tp
    H1 = - ((row1 / n) * math.log(row1 / n, 2) + (row2 / n) * math.log(row2 / n, 2))

    col1 = tn + fp
    col2 = fn + tp
    H2 = - ((col1 / n) * math.log(col1 / n, 2) + (col2 / n) * math.log(col2 / n, 2))

    p00 = 1 if tn == 0 else tn / n
    p01 = 1 if fn == 0 else fn / n
    p10 = 1 if fp == 0 else fp / n
    p11 = 1 if tp == 0 else tp / n
    H12 = - ((tn / n) * math.log(p00, 2) +
             (fn / n) * math.log(p01, 2) +
             (fp / n) * math.log(p10, 2) +
             (tp / n) * math.log(p11, 2))
    MI = H1 + H2 - H12

    VOI = H1 + H2 - 2 * MI

    return VOI


## Interclass correlation
def get_icc(y_pred, y_true):
    y_pred = np.ravel(y_pred.astype(np.float))
    y_true = np.ravel(y_true.astype(np.float))

    mean_y_pred = np.mean(y_pred)
    mean_y_true = np.mean(y_true)

    assert y_true.shape[0] == y_pred.shape[0], "shape does't match"
    n = y_true.shape[0]

    ssw = 0.
    ssb = 0.
    gradmean = (mean_y_pred + mean_y_true) / 2.
    for i in range(n):
        r1 = y_pred[i]
        r2 = y_true[i]
        m = (r1 + r2) / 2

        ssw += pow(r1 - m, 2)
        ssw += pow(r2 - m, 2)
        ssb += pow(m - gradmean, 2)

    ssw = ssw / n
    ssb = ssb / (n - 1) * 2

    icc = (ssb - ssw) / (ssb + ssw)
    return icc


## Probability distance
def get_pbd(y_pred, y_true):
    # y_pred = np.atleast_1d(y_pred.astype(np.bool))
    # y_true = np.atleast_1d(y_true.astype(np.bool))
    y_pred = np.ravel(y_pred.astype(np.float))
    y_true = np.ravel(y_true.astype(np.float))

    assert y_true.shape[0] == y_pred.shape[0], "shape does't match"
    n = y_true.shape[0]

    probability_joint = 0.
    probability_diff = 0.

    for i in range(n):
        r1 = y_pred[i]
        r2 = y_true[i]

        probability_diff += abs(r1 - r2)
        probability_joint += r1 * r2

    pd = -1
    if probability_joint != 0:
        pd = probability_diff / (2 * probability_joint)

    return pd


## Cohen Kappa Coefficient
def get_kap(y_pred, y_true):
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = float(np.count_nonzero(y_pred & y_true))
    tn = float(np.count_nonzero(~y_pred & ~y_true))
    fp = float(np.count_nonzero(y_pred & ~y_true))
    fn = float(np.count_nonzero(~y_pred & y_true))

    agreement = tp + tn

    chance_0 = (tn + fn) * (tn + fp)
    chance_1 = (fp + tp) * (fn + tp)
    chance = chance_0 + chance_1

    sum = (tn + fn + fp + tp)
    chance = chance / sum

    kappa = (agreement - chance) / (sum - chance)
    return kappa


## Area under ROC curve
def get_auc(y_pred, y_true):
    y_pred = np.atleast_1d(y_pred.astype(np.bool))
    y_true = np.atleast_1d(y_true.astype(np.bool))

    tp = float(np.count_nonzero(y_pred & y_true))
    tn = float(np.count_nonzero(~y_pred & ~y_true))
    fp = float(np.count_nonzero(y_pred & ~y_true))
    fn = float(np.count_nonzero(~y_pred & y_true))

    auc = 1 - (fp / (fp + tn) + fn / (fn + tp)) / 2
    return auc


## Hausdorf distance
def get_hd(y_pred, y_true):
    from medpy.metric.binary import hd
    return hd(y_pred, y_true)


## Average surface distance
def get_asd(y_pred, y_true):
    from medpy.metric.binary import asd
    return asd(y_pred, y_true)


## Average symmetric surface distance.
def get_assd(y_pred, y_true):
    from medpy.metric.binary import assd
    return assd(y_pred, y_true)


## Mahalanobis distance
# def mhd(y_pred,y_true):

## peak signal noise ratio
def get_psnr(y_pred, y_true, data_range=None):
    from skimage.metrics import peak_signal_noise_ratio
    return peak_signal_noise_ratio(y_pred, y_true, data_range=data_range)


## mean squared error
def get_mse(y_pred, y_true):
    from skimage.metrics import mean_squared_error
    return mean_squared_error(y_pred, y_true)


## normalized root mean squared error
def get_nrmse(y_pred, y_true):
    from skimage.metrics import normalized_root_mse
    return normalized_root_mse(y_pred, y_true)


## structural similarity
def get_ssim(y_pred, y_true, data_range=None, win_size=None):
    from skimage.metrics import structural_similarity
    return structural_similarity(y_pred, y_true, data_range=data_range, win_size=win_size)


## get kl divergence
def get_kl_div(pk, qk):
    '''
    D_{KL} (p_{k}||q_{k})
    :param pk:
    :param qk:
    Attention: 不能有0
    :return:
    '''
    from scipy.stats import entropy
    from mtools.mimage.mimage import norm_zero_one
    pk = norm_zero_one(pk) * 0.98 + 0.01
    qk = norm_zero_one(qk) * 0.98 + 0.01

    pk = np.asarray(pk).flatten()
    qk = np.asarray(qk).flatten()

    pk = pk / np.sum(pk)
    qk = qk / np.sum(qk)

    kl = entropy(pk, qk)
    return kl


def shot_acc(preds, labels, train_data, many_shot_thr=80, low_shot_thr=60, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def get_chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    from sklearn.neighbors import NearestNeighbors

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def all(y_pred, y_true):
    metric = {
        "dice: ": get_dice(y_pred, y_true),
        "jaccard: ": get_jac(y_pred, y_true),
        "true positive rate: ": get_tpr(y_pred, y_true),
        "true negative rate: ": get_tnr(y_pred, y_true),
        "false positive rate: ": get_fpr(y_pred, y_true),
        "false negative rate: ": get_fnr(y_pred, y_true),
        "recall: ": get_recall(y_pred, y_true),
        "sensitivity: ": get_sensitivity(y_pred, y_true),
        "specificity: ": get_specificity(y_pred, y_true),
        "precision: ": get_precision(y_pred, y_true),
        "accuracy: ": get_accuracy(y_pred, y_true),
        "fall out: ": get_fallout(y_pred, y_true),
        "global consistency error: ": get_gce(y_pred, y_true),
        "volumetric similarity: ": get_vs(y_pred, y_true),
        "rand index: ": get_ri(y_pred, y_true),
        "adjusted rand index: ": get_ari(y_pred, y_true),
        "mutual information: ": get_mi(y_pred, y_true),
        "variation of information: ": get_voi(y_pred, y_true),
        "interclass correlation: ": get_icc(y_pred, y_true),
        "probability distance: ": get_pbd(y_pred, y_true),
        "cohen kappa distance: ": get_kap(y_pred, y_true),
        "area under ROC curve: ": get_auc(y_pred, y_true),
        "hausdorff distance: ": get_hd(y_pred, y_true),
        "average surface distance: ": get_asd(y_pred, y_true),
        "average symmetric surface distance: ": get_assd(y_pred, y_true)
    }
    return metric


predict = [2, 0, 0, 0, 1, 2, 2, 0, 1, 0, 1, 2, 0, 0, 0, 1, 1, 2, 2, 1, 1, 0, 0, 0, 2, 2, 1, 0, 1, 1, 0]
label = [2, 0, 0, 0, 1, 1, 2, 0, 2, 0, 1, 2, 0, 2, 0, 1, 0, 1, 1, 1, 1, 0, 2, 0, 1, 2, 1, 1, 1, 0, 0]

# print(get_multi_accuracy(y_pred=predict,y_true=label))

# from sklearn.metrics import classification_report
# print(classification_report(label,predict))


'''
{
 √ 'accuracy: ': 0.9594577763263282,
 √ 'adjusted rand index: ': 0.582975741594756,
 √ 'area under ROC curve: ': 0.8309073834309437,
 √ 'average surface distance: ': 7.297906084656462,
 √ 'average symmetric surface distance: ': 7.2412979993026845,
 √ 'cohen kappa distance: ': 0.610608166371582,
 √ 'dice: ': 0.6319104211924543,
 √ 'fall out: ': 0.02607139236676581,
 √ 'false negative rate: ': 0.31211384077134663,
 √ 'false positive rate: ': 0.02607139236676581,
 √ 'global consistency error: ': 0.07053121258712024,
 √ 'hausdorff distance: ': 27.0,
 ×√ 'interclass correlation: ': -0.030539043233363098,
 √ 'jaccard: ': 0.4618925770513069,
 √ 'mutual information: ': 0.11488482090495616,
 √ 'precision: ': 0.5843590513359351,
 × 'probability distance: ': 58.206075006421784,
 √ 'rand index: ': 0.9039390205021289,
 √ 'recall: ': 0.6878861592286534,
 √ 'sensitivity: ': 0.6878861592286534,
 √ 'specificity: ': 0.9739286076332342,
 √ 'true negative rate: ': 0.9739286076332342,
 √ 'true positive rate: ': 0.6878861592286534,
 √ 'variation of information: ': 0.3847933259288091,
 √ 'volumetric similarity: ': 0.9186264510700923}
 '''
