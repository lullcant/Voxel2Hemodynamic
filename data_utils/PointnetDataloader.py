import os
from os.path import join as ojoin
import numpy as np
import copy
import time
import random
import csv
import math
import datatable as dt
import pandas as pd
import pickle
import bisect

from tqdm import tqdm
from torch.utils.data import Dataset
from .ffr_utils import *
from sklearn.neighbors import KDTree
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
class ScanDataset_FFR_all_branch():
    def __init__(self, root, centerline_root, block_points=4096, split='test', block_size=32, batch_size=1):
        self.root = root
        self.centerline_root = centerline_root
        self.block_points = block_points
        self.split = split
        self.patch_size = block_size
        self.npoints_threshold = 64  # minimum npoint in a block
        self.base_block_size = 16
        self.single_batch_size = batch_size
        
        ffrs = sorted(os.listdir(self.root))
        ffrs = [ffr for ffr in ffrs if 'ffr' in ffr and ffr.endswith('txt')]
        
        ## set train and test dataset
        split_ind = int(len(ffrs)*0.8)
        self.file_list = ffrs[:split_ind] if split == 'train' else ffrs[:split_ind]
        # self.file_list = ffrs[:split_ind]

        # # ## quick test
        self.file_list = self.file_list[:4]
        # self.file_list = ffrs
        
        # ## for pre-setted tarin and test dataset
        # test_case_id = [16, 17, 18, 19, 36, 37, 38, 39]
        # get_name_id = lambda x : int(x[5:x.index('_')])
        # if split == 'train':
        #     self.file_list = [f for f in ffrs if get_name_id(f) not in test_case_id]
        # else:
        #     self.file_list = [f for f in ffrs if get_name_id(f) in test_case_id]
        
        self.id_nodes = [] # '01' : [xyz, xyz, ...]
        self.centerline_points_dic = []
        self.node_info_dic = []
        self.coor_max_min = []
        self.label_mean = []  # mean ffr
        self.ffr_points_list = []
        self.ffr_labels_list = []
        self.stenosis = []  # stenosis location and branch id
        self.stenosis_branch_tpye = []
        self.name_infos = []
        self.stenosis_infos = []
        
        self.cut_ids = []
        self.cut_locations = []
        
        ## we need to resample centerline
        RESAMPE_SPACING = (0.5, 0.5, 0.5)
        self.branch_ids = ['\x00', '\x01', '\x02', '\x03', '\x04' ,'\x05', '\x06', '\x07', '\x08', '\x09']
        self.main_brnaches = ['LM', 'LAD', 'LCX', 'RCA']
        
        self.map_str = lambda x : "*".join(map(str, x))
        self.map_float = lambda x : list(map(float, x.split('*')))
        self.normalize_array = lambda x : ((x-np.min(x)) / (np.max(x)-np.min(x)))
        file_list_branch_level = []  # record file name
        
      
        ## frequency analysis
        ffr_sample_frequency = np.array([[]])
        
        ## ffr frequency analysis,  1~0.9, 0.8~0.9, 0.8~0.6, 0.6~0.0
        ffr_centernodes_frequency = [[], [], [], []]
        count_ratio = lambda x_arr, a, b : np.around(float(np.where((x_arr > a) & (x_arr <= b))[0].shape[0]) / x_arr.shape[0], 2)
        ffr_idxs = []
        ffr_idxs_branch_id = []  # count specific coronary branch
        ffr_idxs_ffr = []  # record ffr
        
        for ffr_name in tqdm(self.file_list, total=len(self.file_list)):
            
            # if '00184868' not in ffr_name:
            #     continue
            # if '095664' not in ffr_name:
            #     continue
            # if '133.30000015060100281906200016887' not in ffr_name:
            #     continue
            
            ## get centerline
            word1 = 'pointnet_ffr_pressure_surface_all.txt'
            word2 = 'centerline.json'
            centerline = ffr_name.replace(word1, '')[:-4]
            # centerline = ffr_name.replace(word1, '')
            centerline = centerline + word2
            centerline = os.path.join(self.centerline_root, centerline)
            
            ## load data (NOTE: do not use np.loadtxt)
            ffr_path = os.path.join(self.root, ffr_name)
            ffr_data_with_aorta = dt.fread(ffr_path, sep=' ').to_numpy() # x/y/z/ffr/pressure/label/distance/cx1/cy1/cz1/cx2/cy2/cz2
            spacing, stenosis, ffr_data_with_aorta = ffr_data_with_aorta[0], ffr_data_with_aorta[1], ffr_data_with_aorta[2:]
            
            # ## add pesudo pressure
            # spacing = spacing[2:5]
            # ffr_data_with_aorta = np.concatenate((ffr_data_with_aorta[:,:4],  np.zeros((ffr_data_with_aorta.shape[0], 1)), ffr_data_with_aorta[:,4:]), axis=1)
            
            spacing = spacing[3:6]
            stenosis_location, stenosis_branch_type = stenosis[2:5], stenosis[-1]
            
            ## data check
            # 1. aorta ffr should all be 1
            aorta_ffr_mean = np.round(np.mean(ffr_data_with_aorta[np.where(ffr_data_with_aorta[:, 5] == 100)][:, 3]), 2)
            
            if aorta_ffr_mean != 1:
                print("FFR calculation fail...")
                continue
            
            ## load centerline
            proportion = [spacing[i]/RESAMPE_SPACING[i] for i in range(len(RESAMPE_SPACING))]
            json_str = open(centerline)
            path_id_dict, ids, path_id_type = json_load(json_str, proportion)
            json_str = open(centerline)
            stenosis_info = self.get_stenosis_weight(json_str, proportion)
            stenosis_info.extend(list(stenosis_location[::-1]))
            
            ## # add coronary name information
            # if ffr_name != 'image0846600_pointnet_ffr_surface_all.txt': continue
            check_names = self.main_brnaches
            name_info = {}
            for name in check_names:
                result = get_coronary_branch_by_name(name, ids, path_id_type, path_id_dict)
                if result:
                    name_info.update(result)
            
            # ## set pseudo stenosis location
            # sudo_nodes = [self.map_float(n) for n in name_info if name_info[n]['name'] == 2]
            # stenosis_location, stenosis_branch_type = sudo_nodes[len(sudo_nodes)//2], 2

            centerlines_pathes, id_nodes, nodes_id, id_endings = split_centerline(path_id_dict, ids)
            if not len(centerlines_pathes): continue  # single centerline
            centerlines_pathes = [np.array(path).reshape(-1, 3) for path in centerlines_pathes]

            coronary_num = np.unique(ffr_data_with_aorta[:, 5]).shape[0]
            if coronary_num != 4: continue  # number of coronary artery (usually 2)
            for cor_ind in range(1, coronary_num-1):
                ffr_data = ffr_data_with_aorta[ffr_data_with_aorta[:, 5] == cor_ind]
                
                points, ffr, pressure, distance, cen_father_node, cen_node = \
                    ffr_data[:, 0:3], ffr_data[:, 3], ffr_data[:, 4], ffr_data[:, 6], ffr_data[:, 7:10], ffr_data[:, 10:]  # label: 0-background, 100:aorta, others:coronary
                # print(f"Points: {points.shape}")s
                labels = ffr
                if cen_node.shape[-1] != 3 or distance.shape[0] == 0 or labels.shape[0] <= (self.block_points / 2) \
                    or (np.min(distance) == np.max(distance)):
                    continue
                radius = self.get_radius(points, cen_father_node, cen_node)
                # distance = self.normalize_array(distance)[:, None]
                distance = distance[:, None] / 600.
                labels[labels <= 0] = 0
                labels[labels > 1] = 1
                self.label_mean.append(np.mean(labels))
                centerline_ind, matched_ind = self.get_related_centerline(centerlines_pathes, points)
                matched_centerline = centerlines_pathes[centerline_ind]
                
                cur_ids_info = id_nodes[centerline_ind]
                self.id_nodes.append(cur_ids_info)
                
                ## record useful information
                center_point_dic = defaultdict(list)
                node_info = defaultdict(dict)
                ## # record coroanry branch name and location info
                name_location = [] 
                filtered_points = []
                for i, ind in enumerate(matched_ind):
                    ind = ind[0]
                    point = points[i]
                    filtered_points.append(point)
                    center_node = matched_centerline[ind]
                    center_key = self.map_str(center_node)
                    center_point_dic[center_key].append(point)
                    if center_key in name_info:
                        name_location.append(list(name_info[center_key].values()))  # add name info
                    else:
                        name_location.append([0, 0])
                    node_key = self.map_str(point)
                    node_info[node_key]['ffr'] = labels[i]
                    node_info[node_key]['distance'] = distance[i]
                    node_info[node_key]['radius'] = radius[i]
                    node_info[node_key]['cen_father_node'] = cen_father_node[i]
                    node_info[node_key]['cen_node'] = cen_node[i]
                    
                self.centerline_points_dic.append(center_point_dic)
                radius = self.get_min_max_radius(center_point_dic, node_info, points)
                self.node_info_dic.append(node_info)
                self.coor_max_min.append([np.max(points), np.min(points)])
                points_dis = np.concatenate((points, cen_father_node, cen_node, radius, distance, name_location), axis=1)
                # print(f"Points list: {points_dis.shape}")
                # points_dis = []
                self.ffr_points_list.append(points_dis)
                self.ffr_labels_list.append(labels)
                file_list_branch_level.append(f'{ffr_name}_{cor_ind}')
                self.name_infos.append(name_info)
                self.stenosis_infos.append(stenosis_info)
                
        self.file_list = file_list_branch_level
        print(f"file list: {len(self.file_list)}")
        
        random_ind = list(range(len(ffr_idxs)))
        random.shuffle(random_ind)
    
    def __getitem__(self, idx):
        centerline_dic = self.centerline_points_dic[idx]
        id_nodes = self.id_nodes[idx]
        nodes_info = self.node_info_dic[idx]
        amax, amin = self.coor_max_min[idx]
        stenosis_location = self.stenosis_infos[idx]
        whole_data = self.ffr_points_list[idx]
        print(f"Whole data: {whole_data.shape}")
        
        ids = list(id_nodes.keys())
        
        stenosis_nodes = self.get_stenosis_downstream_nodes(id_nodes, centerline_dic, stenosis_location)
        
        selected_points, labels, selected_ind, ending_flag, stenosis_flag = \
            self.get_data(ids, centerline_dic, id_nodes, nodes_info, stenosis_info=stenosis_nodes, ori_data=whole_data)
        # selected_points[...,:3] = 2 * ((selected_points[...,:3] - amin) / (amax -amin)) - 1
        selected_points[...,:3] = selected_points[...,:3] / 600. 
        selected_points = selected_points.reshape(-1, self.block_points, selected_points.shape[-1])
        labels = labels.reshape(-1, self.block_points)
        selected_ind = selected_ind.reshape(-1, self.block_points)
        ending_flag = ending_flag.reshape(-1, self.block_points)
        stenosis_flag = stenosis_flag.reshape(-1, self.block_points)
        # print(f"Infer shape: {selected_points.shape, labels.shape, selected_ind.shape}")
        return selected_points, stenosis_flag, labels, ending_flag, selected_ind
    
    def __len__(self):
        return len(self.name_infos)
    
    
    def get_radius(self, center, pre, after):
        direction = pre - after
        v1 = pre - center
        v2 = after - center
        direction_norm = np.linalg.norm(direction, axis=1)
        direction_norm[direction_norm == 0] = 1
        radius = np.linalg.norm(np.cross(v1, v2, axis=1), axis=1) / direction_norm
        radius_min, radius_max = np.nanmin(radius), np.nanmax(radius)
        radius[np.isnan(radius)] = radius_max
        radius = (radius - radius_min) / (radius_max - radius_min)
        radius = radius.reshape(-1, 1)
        return radius
    
    def get_min_max_radius(self, center_point_dic, node_info, input_points):
        """record min and max radius at each cross section"""
        # min_r, max_r = 1e6, -1
        for key in center_point_dic:
            points = np.array(center_point_dic[key]).reshape(-1, 3)
            center = np.array(list(map(float, key.split('*')))).reshape(-1, 3)
            norm = np.linalg.norm((points - center), axis=1)
            min_norm, max_norm = np.min(norm), np.max(norm)
            # min_r = min_norm if min_norm < min_r else min_r
            # max_r = max_norm if max_norm > max_r else max_r
            for p in points:
                p_key = self.map_str(p)
                node_info[p_key]['radius'] = [min_norm, max_norm]
        radius = []
        min_section_size, max_section_size = 10000, -1
        for p in input_points:
            if len(node_info[self.map_str(p)]) == 0:
                continue
            short_r, long_r = node_info[self.map_str(p)]['radius']
            corss_section_size = math.pi*short_r*long_r
            min_section_size = min(corss_section_size, min_section_size)
            max_section_size = max(corss_section_size, max_section_size)
            node_info[self.map_str(p)]['radius'] = [corss_section_size]
        for p in input_points:
            if len(node_info[self.map_str(p)]) == 0:
                continue
            corss_section_size = node_info[self.map_str(p)]['radius'][0]
            corss_section_size = (corss_section_size - min_section_size) / (max_section_size - min_section_size)
            node_info[self.map_str(p)]['radius'] = [corss_section_size]
            radius.append([corss_section_size])
        return radius
            
    def get_related_centerline(self, centerlines_pathes, points):
        """Map coronary branch and centerline and assign each coronary point
        to a center.

        Args:
            centerlines_pathes ([[]]): list of coronary arteries
            points (ndarray): coronary surface points

        Returns:
            matched_centerline_index: int, mapped centerline ind
            matched_ind: ndarray, kdtree returned mapped ind
        """
        min_radius = 1e6
        matched_centerline_index = 0
        matched_ind = None
        for centerline_index, centerline_array in enumerate(centerlines_pathes):
            kd_tree = KDTree(centerline_array, leaf_size=5)
            kd_dis, kd_ind = kd_tree.query(points, 1)  # search nearest cneterline node
            mean_radius = np.mean(kd_dis)
            if mean_radius < min_radius:
                min_radius = mean_radius
                matched_centerline_index = centerline_index
                matched_ind = kd_ind
        return matched_centerline_index, matched_ind
    
    def get_data(self, ids, centerline_dic, id_nodes, nodes_info, stenosis_info={}, ori_data=[]):
        """Get data along with centerline"""
        selected_points = []
        extracted_info = defaultdict(list)
        keys = ['ffr', 'distance', 'radius', 'cen_father_node', 'cen_node']
        ending_record = []
        stenosis_record = []
        
        tmp = 0
        for k in centerline_dic:
            tmp += len(centerline_dic[k])
        
        ## get ending flag for each center node
        ending_flag = self.get_endings(id_nodes, centerline_dic)
        stenosis_flag = 0
        
        for k in centerline_dic:
            end_flag = 0 if k not in ending_flag else 1
            if k in stenosis_info and stenosis_flag == 0:
                stenosis_flag = 1
            # stenosis_flag = 0 if k not in stenosis_info else 1
            points = centerline_dic[k]
            selected_points.extend(points)
            for p in points:
                p_key = self.map_str(p)
                for k in keys:
                    extracted_info[k].append(nodes_info[p_key][k])
                ending_record.append(end_flag)
                stenosis_record.append(stenosis_flag)
            if stenosis_flag == 1:
                stenosis_flag = -1
        
        selected_num = len(selected_points)
        print(f"Number of points: {selected_num}")
        selected_points = np.array(selected_points)
        # print(f"selected_points: {selected_points.shape}")
        # exit()
        
        # num_batch = int(np.ceil(selected_num/ self.block_points))
        # point_size = int(num_batch * self.block_points)
        # point_idxs = np.array(list(range(selected_num-1)))
        # replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
        # point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
        # selected_ind = np.concatenate((point_idxs, point_idxs_repeat))
        # np.random.shuffle(selected_ind)
        
        # print(f"Block points: {self.block_points}")
        selected_ind = [i for i in range(selected_num)]
        if selected_num >= self.block_points:
            selected_ind = np.random.choice(selected_ind, self.block_points, replace=False)
        else:
            selected_ind = np.random.choice(selected_ind, self.block_points, replace=True)
            
        # np.random.shuffle(selected_ind)
        # print(f"selected_points: {selected_points.shape}")
        selected_points_base = selected_points[selected_ind]
        # print(f"selected_points_base: {selected_points_base.shape}")
        
        ## get entry radius
        entry_radius = extracted_info['radius'][0]
        radius_wave = np.array(extracted_info['radius']) - entry_radius
        
        distance = np.array(extracted_info['distance'])[selected_ind]
        father_node = np.array(extracted_info['cen_father_node'])[selected_ind]
        center_node = np.array(extracted_info['cen_node'])[selected_ind]
        radius = np.array(extracted_info['radius'])[selected_ind]
        radius_wave = radius_wave[selected_ind]
        points = np.concatenate((selected_points_base, distance, radius), axis=1)
        # points = selected_points_base
        labels = np.array(extracted_info['ffr'])[selected_ind]
        # print(f"Labels: {labels.shape, selected_ind.shape}")
        ending_record = np.array(ending_record)[selected_ind]
        stenosis_record = np.array(stenosis_record)[selected_ind]
        
        if len(ori_data):
            ori_cors = np.array(ori_data)[:,:3]
            ori_cors_key = [self.map_str(list(c)) for c in ori_cors]
            map_ind = []
            for c in selected_points:
                map_ind.append(ori_cors_key.index(self.map_str(c)))
            selected_ind = np.array(map_ind)[selected_ind]
        
        return points, labels, selected_ind, ending_record, stenosis_record
    
    def get_endings(self, id_nodes, centerline_dic):
        ids = list(id_nodes.keys())
        ids = sorted(ids, key=lambda x:len(x))[::-1]
        end_ids = []
        for id in ids:
            if not len(end_ids):
                end_ids.append(id)
                continue
            end_flag = True
            for end_id in end_ids:
                if id in end_id:
                    end_flag = False
                    break
            if end_flag:
                end_ids.append(id)
            
        center_ending_flag = {}
        for id in id_nodes:
            if id not in end_ids:
                continue
            center_nodes = id_nodes[id]
            center_nodes = [node for node in center_nodes if len(centerline_dic[self.map_str(node)])]
            if len(center_nodes) <= 100:
                for node in center_nodes:
                    center_ending_flag[self.map_str(node)] = 1
            if len(center_nodes) >= 200:
                for node in center_nodes[-int(len(center_nodes)*0.2):]:
                    center_ending_flag[self.map_str(node)] = 1
            else:
                for node in center_nodes[-100:]:
                    center_ending_flag[self.map_str(node)] = 1
        return center_ending_flag
    
    def get_stenosis_downstream_nodes(self, id_nodes, centerline_dic, stenosis_locations):
        downstream_nodes = {}
        print(f"stenosis_locations: {stenosis_locations}")
        stenosis_locations = np.array(stenosis_locations).reshape(-1, 3)
        for id in id_nodes:
            center_nodes = id_nodes[id]
            center_nodes = np.array([node for node in center_nodes if len(centerline_dic[self.map_str(node)])]).reshape(-1, 3)
            if not len(center_nodes) or not len(stenosis_locations):
                continue
            kd_tree = KDTree(center_nodes)
            kd_dis, kd_ind = kd_tree.query(stenosis_locations)
            for i, dis in enumerate(kd_dis):
                if dis >= 2:
                    continue
                node_id = kd_ind[i][0]
                # print(f"Node id: {node_id, len(center_nodes)}")
                node = center_nodes[node_id]
                # print(f"Dis: {dis, stenosis_locations[i], node}")
                # exit()
                ## make sure it is not endings
                down_ind = min(len(center_nodes) - 1, node_id + 60)
                downstream_node = center_nodes[down_ind]
                downstream_nodes[self.map_str(downstream_node)] = 1
                # print(f"downstream_node: {downstream_node}")
        return downstream_nodes
    
    def get_stenosis_weight(self, json_str, proportion):
        stenosis_info = get_stenosis_info(json_str, proportion)
        if not len(stenosis_info):
            return []
        max_stenosis = stenosis_info['MaxStenosisLocation_Diameter']
        return max_stenosis