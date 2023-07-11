

def get_selected_crops_using_nms(ratios, centers, box_shape, iou_threshold):
    '''
    使用nms选框
    :param ratios:  框的分数（指标越大越好）
    :param centers: 框的中心
    :return:
    '''
    from mtool.mutils.mboxes import get_bbox_IOU
    import numpy as np

    ratios = np.asarray(ratios)
    # print("ratio: {}".format(len(ratios)))

    ## 从大到小的指标进行排序
    ranked_ratios = -np.sort(-ratios)
    ranked_indexs = np.argsort(-ratios)
    # print("rank index num: {}".format(len(ranked_indexs)))

    selected_indexes = [0]
    for index, ratio in enumerate(ranked_ratios[1:], 1):
        ## 遍历之前的框，如果IoU超过阈值，就把这个框舍弃
        flag = False
        for selected in selected_indexes:
            iou = get_bbox_IOU(
                c1=centers[selected],
                c2=centers[index],
                shape=box_shape
            )
            if iou > iou_threshold:
                flag = True
                break

        if flag == True:
            continue

        selected_indexes.append(index)

    results = ranked_indexs[selected_indexes]
    return results


## 将训练得到的标签平衡
def get_balanced_crops(score, crop_centers, crop_shape,
                       upth, loth,
                       pos_iou_th=0.8, neg_iou_th=1e-2,
                       pos_neg_ratio=1.25):
    '''
    得到相对平衡的分割crops，返回的是被选择的crops的索引号
    :param score: crop的指标分数，以肿瘤分割为例，bounding box 中的肿瘤占整个肿瘤的大小为score
    :param crop_centers: crop bounding box 的中心
    :param crop_shape:   crop bounding box 的大小
    :param upth: idnex 指标的上限，超过的认为 positive
    :param loth: index 指标的下限，低于的认为 negative
    :param pos_iou_th: nms 允许的正向框重叠度 default: 80%
    :param neg_iou_th: nms 允许的负向框重叠度 default: 1%
    :param pos_neg_ratio: 正负样本的比例 default: 1.25
    :return: 被选择的框的中心，正样本个数，负样本个数
    '''
    import numpy as np
    import random
    crop_centers = np.asarray(crop_centers)
    score = np.asarray(score)

    assert len(score) == len(crop_centers), "the number between scores and crop bounding box doesn't match"

    ## 找到正负的样本序号和分数
    pos_crops = crop_centers[score > upth]
    neg_crops = crop_centers[score < loth]

    pos_score = score[score > upth]
    neg_score = score[score < loth]

    # print("score  - pos num: {} neg num: {}".format(len(pos_score),len(neg_score)))
    # print("center - pos num: {} neg num: {}".format(len(pos_crops),len(neg_crops)))

    ## 使用nms算法，挑选框，正向的框可以相对多挑，重叠比例可以更大，负的框相对少挑，重叠比例较小
    sel_pos_index = get_selected_crops_using_nms(ratios=pos_score,
                                                 centers=pos_crops,
                                                 box_shape=crop_shape,
                                                 iou_threshold=pos_iou_th)

    ## 注意，对于负样本这里是分数越低越好
    sel_neg_index = get_selected_crops_using_nms(ratios=-neg_score,
                                                 centers=neg_crops,
                                                 box_shape=crop_shape,
                                                 iou_threshold=neg_iou_th)

    neg_num = int(len(sel_pos_index) // pos_neg_ratio)

    if len(sel_neg_index) > neg_num:
        sel_neg_index = random.sample(list(sel_neg_index), neg_num)

    # print("pos num:{} neg:{}".format(len(sel_pos_index), len(sel_neg_index)))

    results = np.asarray(list(pos_crops[sel_pos_index]) + list(neg_crops[sel_neg_index]))
    return results, len(sel_pos_index), len(sel_neg_index)


## 得到K个不相交的子集
def get_k_samples(total_num, K):
    '''
    K折验证随机分成K个
    :param total_num: 样本总数
    :param K: 折数
    :return: [ [] ...] 总共有K个list
    '''
    import random
    num = total_num // K

    ## 所有的样本
    samples = set(range(total_num))

    result = []
    for index in range(K - 1):
        if len(samples) < num:
            break
        tmp = set(random.sample(samples, num))
        result.append(tmp)
        samples = samples - tmp
    result.append(samples)

    # from pprint import pprint
    # pprint(result)
    return result


def get_k_samples_in_order(order, K):
    import numpy as np
    '''
    :param order: index in order
    :param K:
    :return:
    '''
    ## total number of samples
    num = len(order)

    ## index of arrangement
    order = np.asarray(order)
    index = np.argsort(order)

    ## result
    arrangements = []
    for i in range(num // K):
        arrangements.append(list(index[i * K:(i + 1) * K if i != num // K - 1 else num - 1]))

    results = []
    for k in range(K - 1):
        tmp_result = []
        for i in range(num // K):
            tmp_index = np.random.choice(arrangements[i])
            tmp_result.append(tmp_index)
            del (arrangements[i][arrangements[i].index(tmp_index)])
        results.append(tmp_result)

    arrangements = list(eval(str(arrangements).replace('[', '').replace(']', '')))
    results.append(arrangements)
    return results


## 得到K折验证的子集
def get_k_folder_cross_validation_samples(files, K, order=None):
    '''
    :param files: Samples file , such as dicom
    :param K:     K folder cross validation
    :param order: if not None arrangement in specific random order else random order (default: None)
    :return:
    '''
    total_num = len(files)
    if order is None:
        samples = get_k_samples(total_num=total_num, K=K)
    else:
        assert len(order) == total_num, "order num doesn't match the file numer"
        samples = get_k_samples_in_order(order, K)

    result = dict()
    for index in range(K):
        test = [files[s] for s in samples[index]]
        train = list(set(files) - set(test))
        result[index] = {"train": train, "test": test}

    return result


def test_get_k_samples_in_order():
    order = [19, 18, 17, 16, 15, 14, 13, 12, 11, 20, 21, 22, 23, 24, 25, 26, 27]
    samples = get_k_samples_in_order(order, 5)
    # for s in samples:
    #     for i in s:
    #         print(order[i])
    #
    #     print('-----------------------------')


def test_get_k_samples():
    print(get_k_samples(total_num=9, K=3))


def test_get_k_folder_cross_validation_samples():
    from mtool.mio import get_files_name
    files = get_files_name(dire="../../data/Changhai/image")
    result = get_k_folder_cross_validation_samples(files, 5)
    # pprint(result)


def test_get_balanced_crops():
    import numpy as np
    index = np.load("../../test_ratioes.npy")
    centers = np.load("../../test_centers.npy")

    result = get_balanced_crops(index, upth=0.7, loth=0.2, crop_centers=centers, crop_shape=[32, 64, 64])


if __name__ == '__main__':
    test_get_balanced_crops()
    # test_get_k_samples_in_order()
    # test_get_k_samples()
    # test_get_k_folder_cross_validation_samples()
