import os
from os.path import join
import json
import numpy as np
from sklearn.neighbors import KDTree
import datatable as dt
import tqdm
from tqdm import tqdm
import copy
from scipy import interpolate
from scipy.special import comb
import pandas as pd
import SimpleITK as sitk
from edt3d.edt3d import edt3d_cuda


id_name_map = {
    'LM': 1,
    'LAD': 2,
    'LCX': 3,
    'RCA': 4
}

def get_coronary_name_map(data):
    """get the relationship between id and coronary name
    @data (json)

    return:
        dictionary: {1: LCx, ...}
    """
    id_name = {}
    coronary_branches = data['CoronaryBranches']
    for plaque in coronary_branches:
        id_name[plaque['type']] = plaque['name']
    return id_name

def get_b_spline_3d(points, N):
    ## points 不能存在相同的点
    points = np.asarray(points).copy()
    _, index, counts = np.unique(points, axis=0, return_index=True, return_counts=True)
    index = list(set([i for i in range(len(points))]) - set(index))
    points = np.delete(points, index, axis=0)
    if points.shape[0] == 1:
        return False, None

    if len(points) <= 3:
        points = get_bessel_spline(points, N=6)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    tck, u = interpolate.splprep([x, y, z], k=3, s=0)
    u_fine = np.linspace(0, 1, N, endpoint=True)
    bspline = interpolate.splev(u_fine, tck)
    bspline = np.asarray(bspline).transpose(1, 0)
    return True, bspline

def json_load(json_str, proportion):
    """
    """
    data = json.load(json_str)
    id_name = get_coronary_name_map(data)
    ids = []

    path_id_dict = {}  # id: path_list
    path_id_type = {}  # id: coronary type
    for j in range(len(data['PipeNodes'])):
        path = data['PipeNodes'][j]['Path']
        path_id = data['PipeNodes'][j]['Id']
        path_type = id_name.get(data['PipeNodes'][j]['CoronaryType'], None)
        if len(path) < 1:
            continue
        ids.append(path_id)
        path_list_one = []
        # print(path)
        for i in range(int(len(path)/3)):
            path_x = path[3*i] * proportion[0]
            path_y = path[3*i+1] * proportion[1]
            path_z = path[3*i+2] * proportion[2]

            path_list_one.append([path_x, path_y, path_z])  #  按照zyx顺序读取
        
        path_id_dict[path_id] = path_list_one
        path_id_type[path_id] = path_type
    
    ## B spline
    all_ids = copy.deepcopy(ids)
    all_ids = sorted(all_ids, key=lambda x:len(x), reverse=True)
    for id in all_ids:
        cur_path = path_id_dict[id]
        if len(id) >= 2:
            father_id = id[:-1]
        else:
            continue
        if father_id in path_id_dict:
            cur_path_plus = path_id_dict[father_id][-1:] + cur_path
        if len(cur_path_plus) <= 1:
            continue
        cur_path_plus = np.array(cur_path_plus).reshape(-1, 3)
        flag, interpolated = get_b_spline_3d(cur_path_plus, cur_path_plus.shape[0]*20)
        cur_path_inter = list(interpolated) if flag else cur_path
        # cur_path_inter = cur_path
        path_id_dict[id] = cur_path_inter
    return path_id_dict, ids, path_id_type
def split_centerline(path_id_dict, ids):
    """Split centerline and save seperately
    """
    ids_len = [len(id) for id in ids]
    root_len = min(ids_len)
    root_ids = [ids[i] for i in range(len(ids)) if ids_len[i] == root_len]
    centerlines_pathes = []
    node_id = []  # recorde each nodes id
    id_nodes = []  # recorde id related path
    id_endings = []  # record if this id is ending
    if len(root_ids) != 2:
        return centerlines_pathes, id_nodes, node_id, id_endings
    for root_id in root_ids:
        path = []
        node_id_dic = {}
        id_nodes_dic = {}
        id_endings_list = []
        id_canditates = [root_id]
        while len(id_canditates):
            cur_id = id_canditates[0]
            currrent_path = path_id_dict[cur_id]
            id_nodes_dic[cur_id] = currrent_path
            for n in currrent_path:
                key = "*".join(map(str, n))
                node_id_dic[key] = cur_id

            path.extend(currrent_path)
            del id_canditates[0]
            
            ending_flag = True
            for next_id in ['\x00', '\x01', '\x02', '\x03', '\x04' ,'\x05', '\x06', '\x07', '\x08', '\x09']:
                if cur_id + next_id in ids:
                    id_canditates.append(cur_id + next_id)
                    ending_flag = False
            if ending_flag:
                id_endings_list.append(cur_id)
        centerlines_pathes.append(path)
        node_id.append(node_id_dic)
        id_nodes.append(id_nodes_dic)
        id_endings.append(id_endings_list)
    return centerlines_pathes, id_nodes, node_id, id_endings
def get_coronary_branch_by_name(name, ids, path_id_type, path_id_dict):
    """get coronary branch by name and record its name and location

    Args:
        name (str): coronary artery name
        ids (list): list of id
        path_id_type (list): list of coronary segments
        path_id_dict (dict): segment id and its name

    Returns:
        dict: {'x*y*z': {'name': 1, 'location': 1}, ...}
    """
    assert (name in ['LM', 'LAD', 'LCX', 'RCA'])
    result = {}
    map_str = lambda x:"*".join(map(str, x))
    target_ids = [the_id for the_id in ids if path_id_type.get(the_id) == name]
    if not len(target_ids):
        return False
    target_ids = sorted(target_ids, key=lambda x:len(x))
    # print(f"For coronary artery {name}, there are ids: {target_ids}")
    target_path = []
    for the_id in target_ids:
        target_path.extend(path_id_dict[the_id])
    # print(f"There are total {len(target_path)} nodes")

    path_array = np.array(target_path)
    path_norm_arr = np.linalg.norm((path_array[:-1] - path_array[1:]), axis=-1).squeeze()
    path_norm = np.sum(path_norm_arr)

    ## cut endings
    if name != 'LM':
        len_p = len(target_path)
        if len_p <= 10 and len_p >= 5:
            target_path = target_path[:-2]
        elif len_p > 10:
            cut_len = path_norm * 0.15  # cut 15%
            cutted = 0
            for i in range(len_p-1):
                cutted += path_norm_arr[len_p-2-i]
                if cutted >= cut_len:
                    target_path = target_path[:-i]
                    break
        # print(f"{name} original vs cut: {len_p, len(target_path)}")

    ## update path norm
    path_array = np.array(target_path)
    path_norm_arr = np.linalg.norm((path_array[:-1] - path_array[1:]), axis=-1).squeeze()
    path_norm = np.sum(path_norm_arr)

    split_times = 1
    if name == 'LM':
        split_times = 1
    elif name == 'LCX':
        split_times = 2
    else:
        split_times = 3
    split_len = path_norm // split_times
    cutted = 0
    start_id = 0
    for j in range(len(target_path)-2):
        key = map_str(target_path[j])
        location_id = 3 if (start_id == 1 and name == 'LCX') else start_id+1
        result[key] = {'name': id_name_map[name], 'location': location_id}
        cutted += path_norm_arr[j]
        if cutted >= split_len:
            cutted = 0
            start_id += 1

    # ###### Based on number of node method. It is not accurate considering different node density.
    # ## cut endings
    # if name != 'LM':
    #     len_p = len(target_path)
    #     if len_p <= 10 and len_p >= 5:
    #         target_path = target_path[:-2]
    #     elif len_p > 10:
    #         target_path = target_path[:int(len_p*0.5)]
    #     print(f"Original vs cut: {len_p, len(target_path)}")

    # split_times = 1
    # if name == 'LM':
    #     split_times = 1
    # elif name == 'LCX':
    #     split_times = 2
    # else:
    #     split_times = 3
    # split_ind = len(target_path) // split_times
    # for i in range(split_times):
    #     cur_segment = target_path[i*split_ind:(i+1)*split_ind]
    #     for cor in cur_segment:
    #         key = map_str(cor)
    #         location_id = 3 if (i == 1 and name == 'LCX') else i+1
    #         result[key] = {'name': id_name_map[name], 'location': location_id}
    return result