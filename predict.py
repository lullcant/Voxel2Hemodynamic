
import argparse
import os
from data_utils.PointnetDataloader import ScanDataset_FFR_all_branch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import pandas as pd
import csv
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import bisect
from mtools.mio import get_json, get_medical_image, save_medical_image, save_csv, get_yaml, save_mps
from mtools.mimage.mbinary import get_largest_n_connected_region
from mtools.mimage.mimage import norm_zero_one
from mtools.mtorch.mtrainer import TrainerBase
import torch.nn.functional as F
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pprint import pprint
import trimesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--block_size', type=int, default=32, help='sample block size [default: 32*2]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--split', type=str, default='test', help='train or test [default: test]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=1, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')
    parser.add_argument('--features', type=int, default=4, help='feature dimension')
    parser.add_argument("-c", "--config", type=str, default="./config-predict.yaml", help="config file",
                        required=False)
    return parser.parse_args()

def add_vote_ffr(vote_label_pool, vote_label_num, point_idx, pred_label, weight, top_branch=False, record_diff=[]):
    # if top_branch:
    #     pred_label = pred_label[:, :1360]
    #     point_idx = point_idx[:, :1360]
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            cur_pred_val = float(pred_label[b, n])
            vote_label_pool[int(point_idx[b, n]), 0] += cur_pred_val
            vote_label_num[int(point_idx[b, n]), 0] += 1
            # if cur_pred_val >= 0.1:
            #     pre = 0
            #     if vote_label_num[int(point_idx[b, n]), 0] != 0:
            #         pre = vote_label_pool[int(point_idx[b, n]), 0] / vote_label_num[int(point_idx[b, n]), 0]
            #         record_diff.append(round(pre-cur_pred_val, 2))
            #     vote_label_pool[int(point_idx[b, n]), 0] += cur_pred_val
            #     vote_label_num[int(point_idx[b, n]), 0] += 1
    return vote_label_pool, vote_label_num, record_diff
class Trainer(TrainerBase):

    def predict(self, data):
        '''
        验证过程中预测
        :param data: [for index, data in tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader))]
        :return: pred, labels [lists]
        '''
        images = data
        if self.use_cuda:
            images = images.cuda()

        with torch.no_grad():
            predict, _ = self.model['Unet'](images)
            # reconst = get_largest_n_connected_region(np.asarray(predict.cpu().numpy()[0][0] > 0.5).astype(int), n=3)[0]
            # images = images * (torch.ones(images.size()).cuda() - predict  + torch.from_numpy(reconst[np.newaxis, np.newaxis, :, :, :]).cuda())

            # predict, img_feat = self.model['Unet'](images)
            # reconst = get_largest_n_connected_region(np.asarray(predict.cpu().numpy()[0][0] > 0.5).astype(int), n=3)[1]
            # images = images *  (torch.ones(images.size()).cuda() - predict  + torch.from_numpy(reconst[np.newaxis, np.newaxis, :, :, :]).cuda())

            reconst, img_feat = self.model['Unet'](images)

            # unpools = []
            # for feat in list(reversed(img_feat)):
            #     unpools.append(x.expand(feat.size()) * feat)
            #     x = F.avg_pool3d(x, kernel_size=(2, 2, 2), stride=(2, 2, 2))
            # unpools = list(reversed(unpools))
            # p_verts, p_faces = self.model['Gseg'](unpools)

            # print("1: ", [torch.any(torch.isnan(feat)) for feat in img_feat])
            p_verts, p_faces = self.model['Gseg'](img_feat)
            # print("2: ", [torch.any(torch.isnan(v)) for v in p_verts])

        # predict = list(map(int, list(predict.numpy().argmax(axis=1))))
        return [reconst, p_verts, p_faces]
    
def main(args):
    
    def log_string(str):
        logger.info(str)
        print(str)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # Setting for the segmentation and vectorization

    config = get_yaml(args.config)
    trainer = Trainer(config=config)
    trainer.init_network()
    json = './data.json'
    dire = './data/mcrops'
    info = get_json(json)

    # Setting for the hemodynamic prediction
    NUM_CLASSES = 1
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    BLOCK_SIZE = args.block_size

    data_name = 'test_data'
    root_folder = f'{data_name}'  
    root = f'/data/{root_folder}/'
    centerline_root = f'/data/{data_name}/centerlines'

    # generate the point cloud from dcm

    points = []

    for idx, item in tqdm(enumerate(info[list(info.keys())[-2]]), total=len(info[list(info.keys())[-2]])):
        image, param = get_medical_image(os.path.join(dire, item['image']))
        image = norm_zero_one(image, span=[-200, 400])[np.newaxis, np.newaxis, :, :, :]
        image = torch.from_numpy(image)

        reconst, p_verts, p_faces = trainer.predict(image)
        # reconst = trainer.predict(image)

        # p_verts = p_verts[-1]
        # p_faces = p_faces[-1]

        # reconst = reconst.detach().cpu().numpy()[0][0]
        verts = p_verts[-1].detach().cpu().numpy()[0]
        faces = p_faces[-1].detach().cpu().numpy()[0]


        p_verts = sample_points_from_meshes(Meshes(verts=p_verts[-1], faces=p_faces[-1]),num_samples=20)[0].detach().cpu().numpy()
        p_verts = p_verts * 8 + item['center']

        points.append(p_verts)

    points = np.concatenate(points,axis=0)
    save_csv('./data/test_data/point.txt', points, delimiter=' ')
    
    TEST_DATASET_WHOLE_SCENE = ScanDataset_FFR_all_branch(root, centerline_root, split=args.split, block_points=NUM_POINT, block_size=BLOCK_SIZE)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES, features=args.features).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier = DDP(classifier, device_ids=[args.local_rank], output_device=args.local_rank)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    test_single_segment = False  # result will be saved at segment level
    result_dic = defaultdict(list)  # gt:diff, 2 decimal
    result_distance_analysis = defaultdict(list)
    distance_gap = [0.1 * i for i in range(1, 11)]
    result_record = []
    stenosis_result_record = []

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)
        log_string('---- EVALUATION WHOLE SCENE----')
        pred_mess = []
        record_diffs = []
        for batch_idx in range(num_batches):
            scene_id_batch = scene_id[batch_idx]
            print(f"Infering: {scene_id_batch}")
            pred_pools = []
            gt_pools = []
            diff_pools = []
            gt_label_pools = []
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.ffr_points_list[batch_idx]
            # print(f"whole_scene_data: {whole_scene_data.shape}")
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.ffr_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            vote_label_num = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_stenosis_flag, scene_label, scene_ending_flag, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, args.features))
                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_ending_flag = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_stenosis_flag = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))
                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_ending_flag[0:real_batch_size, ...] = scene_ending_flag[start_idx:end_idx, ...]
                    batch_stenosis_flag[0:real_batch_size, ...] = scene_stenosis_flag[start_idx:end_idx, ...]
                    # batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().numpy()[...,0]
                    
                    ## save tmp result
                    # print(f"batch_pred_label: {batch_pred_label.shape}")
                    columns = ['x', 'y', 'z', 'ffr', 'pred', 'diff']
                    diff = np.abs(batch_label[...,None] - batch_pred_label[...,None])
                    non_ending_ind = np.where(batch_ending_flag == 0)
                    out_data = batch_data[...,:3][non_ending_ind]
                    out_label = batch_label[...,None][non_ending_ind]
                    out_pred = batch_pred_label[...,None][non_ending_ind]
                    diff = diff[non_ending_ind]
                    batch_stenosis_flag = batch_stenosis_flag[...,None][non_ending_ind]
                    ## check downstream stenosis
                    # out_label[np.where(batch_stenosis_flag == 1)] = 2
                    # print(f"Flitered: {out_data.shape, out_label.shape, out_pred.shape, diff.shape}")
                    # tmp_data = np.concatenate((batch_data[...,:3], batch_label[...,None], batch_pred_label[...,None], diff), axis=-1)
                    tmp_data = np.concatenate((out_data, out_label, out_pred, diff), axis=-1)
                    # df = pd.DataFrame(tmp_data, columns=columns)
                    # result_filename = os.path.join(visual_dir, f'{scene_id_batch}_result.csv')
                    # df.to_csv(result_filename, index=False)
                    mae = np.nanmean(diff)
                    result_record.append([f'{scene_id_batch}_result.csv', np.nanmean(out_label), mae])
                    
                    ## record stenosis information
                    stenosis_ind = np.where(batch_stenosis_flag == 1)
                    if stenosis_ind[0].shape[0]:
                        # print(f"stenosis_ind: {stenosis_ind}")
                        s_label = np.mean(out_label[stenosis_ind])
                        s_pred = np.mean(out_pred[stenosis_ind])
                        s_diff = np.abs(s_label - s_pred)
                        print(f"Stenosis compare: {s_label, s_pred, s_diff}")
                        stenosis_result_record.append([s_label, s_pred, s_diff])
                    # continue
                    # exit()
                    
                    # save all segment prediction to a pool and show all of them
                    pred_output = batch_pred_label.reshape(BATCH_SIZE, NUM_POINT)
                    pred_pools.append(pred_output)

                    cur_gt = whole_scene_data[batch_point_index.astype(np.int64)].squeeze()
                    cur_gt_label = whole_scene_label[batch_point_index.astype(np.int64)].squeeze()
                    cur_diff = pred_output - cur_gt_label[:,None]
                    gt_pools.append(cur_gt)
                    diff_pools.append(cur_diff)
                    gt_label_pools.append(cur_gt_label)

                    vote_label_pool, vote_label_num, record_diffs = add_vote_ffr(vote_label_pool, vote_label_num, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...],
                                               record_diff=record_diffs)
            
            # plt.scatter(record_diffs, [i for i in range(len(record_diffs))])
            # plt.savefig('./temp_result/patch_diff.png')
            # exit()
            # continue

            pred_label = vote_label_pool / vote_label_num
            pred_gt_diff = np.abs(pred_label - whole_scene_label[:,None])
            pred_mse = np.nanmean(pred_gt_diff)
            pred_mess.append(pred_mse)
            print("Pred mse: {}".format(pred_mse))

            print('----------------------------')

            if not test_single_segment:
                result_filename = os.path.join(visual_dir, scene_id[batch_idx] + f'_result.csv')
                columns = ['x', 'y', 'z', 'size', 'distance', 'ffr', 'pred', 'diff']
                out = np.concatenate((whole_scene_data[..., :3], whole_scene_data[..., -4:-2], \
                    whole_scene_label[..., None], pred_label, pred_gt_diff), axis=1)
                out_nan = out[~np.isnan(out).any(axis=1), :]
                df = pd.DataFrame(out_nan, columns=columns)
                df.to_csv(result_filename, index=False)
            else:
                for pool_id, cur_pred_out in enumerate(pred_pools):
                    filename = os.path.join(visual_dir, scene_id[batch_idx] + f'_{pool_id}.txt')
                    gt_csv = os.path.join(visual_dir, scene_id[batch_idx] + f'_gt_{pool_id}.csv')
                    pred_csv = os.path.join(visual_dir, scene_id[batch_idx] + f'_pred_{pool_id}.csv')
                    pred_gt_diff_csv = os.path.join(visual_dir, scene_id[batch_idx] + f'_diff_{pool_id}.csv')
                    # with open(filename, 'w') as pl_save:
                    #     for i in pred_label:
                    #         pl_save.write(str(i) + '\n')
                    #     pl_save.close()
                    
                    cur_gt = gt_pools[pool_id]
                    # print(f"cur gt: {max(np.round(cur_gt.flatten().squeeze(), 2)), min(np.round(cur_gt.flatten().squeeze(), 2))}")
                    cur_diff = diff_pools[pool_id]
                    cur_label = gt_label_pools[pool_id]

                    for g, d in zip(np.round(cur_label.flatten().squeeze(), 2), np.round(cur_diff.flatten().squeeze(), 2)):
                        result_dic[g].append(d)

                    # distance analysis
                    cur_dis = cur_gt[:, -1].flatten().squeeze()
                    cur_dis_min, cur_dis_max = np.min(cur_dis), np.max(cur_dis)
                    cur_dis = (cur_dis - cur_dis_min) / (cur_dis_max - cur_dis_min)
                    for dis, dif in zip(np.round(cur_dis, 2), np.round(cur_diff.flatten().squeeze(), 2)):
                        _gap_id = bisect.bisect(distance_gap, dis)
                        if _gap_id == len(distance_gap):
                            _gap_id -= 1
                        result_distance_analysis[distance_gap[_gap_id]].append(dif)
                    # result_distance_analysis.append([cur_dis, cur_diff.flatten().squeeze()])

                    cur_diff = np.abs(cur_diff)
            if args.visual:
                fout.close()
                fout_gt.close()
        print("Done!")
            
        result_record = sorted(result_record, key=lambda x:x[-1])
        df = pd.DataFrame(result_record, columns=['id', 'mean_ffr', 'mae'])
        df.to_csv(f'./temp_result/{args.log_dir}_{root_folder}.csv', index=False)
        
        stenosis_result_record = sorted(stenosis_result_record, key=lambda x:x[-1])
        df = pd.DataFrame(stenosis_result_record, columns=['label', 'pred', 'diff'])
        df.to_csv(f'./temp_result/{args.log_dir}_{root_folder}_STENOSIS.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
