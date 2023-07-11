from mtools.meval import get_multi_recall, get_multi_precision, get_multi_f1, get_multi_accuracy, shot_acc, get_dice
from mtools.mtorch.mscheduler import CustomizeScheduledOptim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from mtools.mio import source_import
from pprint import pprint, pformat
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import traceback
import logging
import torch
import time
import copy
import os


## 使用kaming对网络进行初始化
def init_net_kaming(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


## 使用Normal对网络进行初始化
def init_net_normal(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


## 使用SGD优化器
def init_optim_SGD(module, param):
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    optimizer = torch.optim.SGD(
        module.parameters(), lr=param['lr'], momentum=momentum, weight_decay=weight_decay)
    return optimizer


## 使用Adam优化器
def init_optim_Adam(module, param):
    optimizer = torch.optim.Adam(module.parameters(), lr=param['lr'])
    return optimizer


## 使用ReduceLROnPlateau学习器
def init_scheder_ReduceLROnPlateau(optimizer, param):
    mode = param['mode']
    factor = param['factor']
    patience = param['patience']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, verbose=True)
    return scheduler


## 使用CosineAnnealingLR学习器
def init_scheder_CosineAnnealingLR(optimizer, param):
    ## 学习率迭代周期的一般
    half_cycle = param['half_cycle']
    ## 学习率的最小值
    eta_min = param['eta_min']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=half_cycle, eta_min=eta_min)
    return scheduler


## 使用自定义学习器
def init_scheder_CustomizeLR(optimizer, param):
    scheduler = CustomizeScheduledOptim(optimizer, T_max=90, eta_min=0.01)
    return scheduler


## Trainer Base (Modules)
class TrainerBase(object):

    def __init__(self, config):
        self.config = config

    ## 初始化日志
    def init_loggers(self):
        LOG_FORMAT = "%(asctime)s  [%(levelname)s]:  %(message)s"
        DAT_FORMAT = "%Y-%m-%d %H:%M:%S"

        logfile = "{}.log".format(self.tag)
        ##pytyhon 3.9可用
        # logging.basicConfig(filename=logfile, level=logging.INFO, encoding='utf-8',
        #                     format=LOG_FORMAT, datefmt=DAT_FORMAT)

        F = open(logfile, encoding="utf-8", mode="a")
        logging.basicConfig(stream=F, level=logging.INFO, format=LOG_FORMAT, datefmt=DAT_FORMAT)

    def init_tfboard(self):
        self.tboard_writer = SummaryWriter()

    ## 加载数据集
    def load_dataset(self):
        '''
        根据配置文件进行数据集加载
        '''
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '----------------  Data Loading ----------------\n'
            '-----------------------------------------------\n'
        )

        ## 数据集参数
        params = self.config['dataset']

        ## 得到训练集和测试集
        dataset_param = list(params['defin_parm'].values())
        train_dataset, valid_dataset = source_import(params['defin_path']).get_dataset(*dataset_param)

        train_nums = train_dataset.__len__()
        valid_nums = valid_dataset.__len__()

        ## 加载数据集到dataloader
        batch_size = params['batch_size']
        num_worker = params['num_worker']
        isdroplast = params['isdroplast']
        is_shuffle = params['is_shuffle']

        ## 暂时不考虑sampler的情况
        defin_sampler = params['defin_sampler']
        param_sampler = list(params['param_sampler'].values())

        self.show_outputs(
            'Batch  size:{}\n'
            'Worker num:{}\n'
            'Train  num:{}\n'
            "Valid  num:{}\n"
            "Drop last :{}\n"
            "Shuffle: {}\n"
                .format(batch_size, num_worker, train_nums, valid_nums, isdroplast, is_shuffle)
        )

        if defin_sampler is not None:
            sampler = source_import(defin_sampler).get_sampler(*param_sampler)(train_dataset)
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker,
                                               sampler=sampler, drop_last=isdroplast)
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker,
                                               shuffle=is_shuffle, drop_last=isdroplast)

        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=False)

    ## 初始化神经网络模块 (module)
    def init_parames(self, method, module):
        if method == 'kaiming':
            init_net_kaming(module=module)
        elif method == 'normal':
            init_net_normal(module=module)
        else:
            assert 'No Initialize Method'
        self.show_outputs("\nModule: {} loaded ! Init Method:{}".format(module._get_name(), method))

    def load_parames(self, checkpoint, module, key):
        module.load_state_dict(torch.load(checkpoint)['networker'][key])
        self.show_outputs("\nModule: {} Checkpoint: {} loaded !".format(module._get_name(), checkpoint))

    ## 初始化神经网络
    def init_network(self):
        self.show_outputs(
            '\n-----------------------------------------------\n'
            '----------------  Model Loading ---------------\n'
            '-----------------------------------------------\n'
        )

        ## 神经网络参数
        params = self.config['network']
        self.use_cuda = (torch.cuda.is_available() and params['use_cuda'])
        self.use_parallel = (self.use_cuda and torch.cuda.device_count() > 1 and params['use_parallel'])

        def load_module(module, param):
            curr_params = param['cur_params'] if 'cur_params' in param else None
            init_method = param['int_method'] if 'int_method' in param else None
            if curr_params is not None:
                self.load_parames(checkpoint=curr_params, module=module, key=key)
            elif init_method is not None:
                self.init_parames(method=init_method, module=module)
            else:
                assert 'No initialization method for module !'

            ## 将模型送入GPU
            if self.use_parallel:
                module = nn.DataParallel(module)

            if self.use_cuda:
                module = module.cuda()
            return module

        ## 神经网络主干
        self.model = dict()
        modules_param = params['modules']
        for key, item in modules_param.items():
            ## 加载网络 - 初始化/装载网络参数
            module = source_import(item['defin_path']).create_model(*list(item['defin_parm'].values()))
            self.model[key] = load_module(module, param=item)

        ## 损失函数
        self.criterions = dict()
        if 'criterions' in params:
            criterions_param = params['criterions']
            for key, item in criterions_param.items():
                criterion = source_import(item['defin_path']).create_loss(*list(item['defin_parm'].values()))
                self.criterions[key] = [load_module(criterion, param=item), item['weight']]

        ## 显示输出
        self.show_outputs(
            'Model Structure:\n{}\n'
            'Criterion:\n{}\n'
            'Use CUDA:      {}\n'
            "Use Parallel:  {}\n"
                .format([s for s in self.model.values()], [s for s in self.criterions.values()], self.use_cuda,
                        self.use_parallel)
        )

    ## 初始化优化器
    def init_optimer(self):
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '--------------  Optimizer Loading -------------\n'
            '-----------------------------------------------\n'
        )

        ## 必须先初始化神经网络
        assert hasattr(self, 'model')
        params = self.config['network']

        self.optimizer = dict()
        for key, module in self.model.items():
            if 'optimizers' in params['modules'][key]:
                optim_param = params['modules'][key]['optimizers']
                if optim_param['type'] == 'Adam':
                    self.optimizer[key] = init_optim_Adam(module, optim_param)
                elif optim_param['type'] == 'SGD':
                    self.optimizer[key] = init_optim_SGD(module, optim_param)
                else:
                    assert 'No recognized optimizer!'

                ## 如果有之前训练的参数，加载参数
                if 'cur_params' in optim_param and optim_param['cur_params'] is not None:
                    self.optimizer[key].load_state_dict(torch.load(optim_param['cur_params'])['optimizer'][key])

                    if 'lr' in optim_param:
                        self.optimizer[key].param_groups[0]['lr'] = optim_param['lr']

                    self.show_outputs("Optimizer: {} Checkpoint: {} loaded !\n".format(key, optim_param['cur_params']))

        for key, optim in self.optimizer.items():
            self.show_outputs(
                'Module: {} \nOptimizer:\n{}\n'
                    .format(key, optim)
            )

    ## 初始化学习器
    def init_scheder(self):
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '--------------  Scheduler Loading -------------\n'
            '-----------------------------------------------\n'
        )

        ## 必须先初始化优化器
        assert hasattr(self, 'optimizer')
        params = self.config['network']

        self.scheduler = dict()
        for key, optim in self.optimizer.items():
            if 'schedulers' in params['modules'][key]:
                sched_param = params['modules'][key]['schedulers']
                if sched_param['type'] == 'CosineAnnealingLR':
                    self.scheduler[key] = [sched_param['type'], init_scheder_CosineAnnealingLR(optim, sched_param)]
                elif sched_param['type'] == 'ReduceLROnPlateau':
                    self.scheduler[key] = [sched_param['type'], init_scheder_ReduceLROnPlateau(optim, sched_param)]
                elif sched_param['type'] == 'CustomizeLR':
                    self.scheduler[key] = [sched_param['type'], init_scheder_CustomizeLR(optim, sched_param)]
                else:
                    assert 'No recognized scheduler!'

                ## 如果有之前训练的参数，加载参数
                if 'cur_params' in sched_param and sched_param['cur_params'] is not None:
                    self.scheduler[key].load_state_dict(torch.load(sched_param['cur_params'])['scheduler'][key])
                    self.show_outputs("Scheduler: {} Checkpoint: {} loaded !\n".format(key, sched_param['cur_params']))

        for key, (type, sched) in self.scheduler.items():
            self.show_outputs(
                'Module: {} \nScheduler: {} \n{}\n'
                    .format(key, type, pformat(sched.state_dict()))
            )

    ## 初始化训练器
    def init_trainer(self):
        ## 初始化日志
        self.use_logger = self.config['monitor']['logger']
        self.use_pprint = self.config['monitor']['stdstream']
        self.use_tboard = self.config['monitor']['tensorboardx']

        self.tag = self.config['tag']
        if self.use_logger: self.init_loggers()
        if self.use_tboard: self.init_tfboard()
        self.show_outputs("\nconfig:\n{}\n\n".format(pformat(self.config)))

        self.load_dataset()  ## 加载数据集
        self.init_network()  ## 初始化神经网络
        self.init_optimer()  ## 初始化优化器
        self.init_scheder()  ## 初始化策略器

        ## 初始化训练参数
        self.checkpoint_mode = self.config['trainer']['checkpoint_mode']  ## cp 保存的模式
        self.validation_step = self.config['trainer']['validation_step']  ## 验证 / 保存的步长
        self.cpnt_dire = self.checkpoint_mode['dire']
        self.save_mode = self.checkpoint_mode['type']

        self.thresh_gradnt = self.config['trainer']['threshold_grad']  ## 梯度裁剪
        self.total_epoches = self.config['trainer']['total_epoches']  ## 训练流程全部的epoch
        self.current_epoch = self.config['trainer']['current_epoch']  ## 训练流程当前的epoch

    ## 保存训练器
    def save_trainer(self, path):
        ## 神经网络模型参数
        os.makedirs(self.cpnt_dire, exist_ok=True)

        model_state_dict = dict()
        for key, module in self.model.items():
            if self.use_parallel: module = module.module
            model_state_dict[key] = copy.deepcopy(module.state_dict())

        ## 优化器参数
        optim_state_dict = dict()
        for key, optim in self.optimizer.items():
            optim_state_dict[key] = copy.deepcopy(optim.state_dict())

        ## 学习器参数
        sched_state_dict = dict()
        for key, (type, sched) in self.scheduler.items():
            sched_state_dict[key] = copy.deepcopy(sched.state_dict())

        torch.save(
            {"networker": model_state_dict,
             "optimizer": optim_state_dict,
             "scheduler": sched_state_dict,
             "cur_epoch": self.current_epoch,
             "best_epch": self.best_epoch,
             "best_indx": self.best_index
             },
            path
        )
        self.show_outputs("\ncheckpoint path: {}\nsave checkpoint done!\n".format(path))

    ## 前向传播，需要重构
    def forward(self, index, data):
        '''
        训练中前向传播的过程，用在一个epoch中的每一批batch中
        :param data: [for index, data in enumerate(self.train_dataloader)]
        :return: predict, label
        '''
        raise NotImplementedError

    ## 计算损失
    def get_loss(self, results):
        loss = 0
        predict, labels = results
        for key, (criterion, weight) in self.criterions.items():
            loss += criterion(predict, labels) * weight
        return loss

    ## 反向传参，需要重构
    def backward(self, loss):
        for optimizer in self.optimizer.values():
            optimizer.zero_grad()

        loss = loss.mean()  # for parallel
        loss.backward()

        ## clip the gradient
        if self.thresh_gradnt is not None:
            for module in self.model.values():
                nn.utils.clip_grad_norm_(module.parameters(), self.thresh_gradnt)

        for key, optimizer in self.optimizer.items():
            optimizer.step()
        return loss.item()

    ## 分批次进行训练，可能需要重构
    def batrain(self, index, data):
        results = self.forward(index=index, data=data)
        loss = self.get_loss(results=results)

        pred, true = results
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        return self.backward(loss), pred, true

    ## 训练器进行训练
    def trainer(self):
        self.show_outputs(
            '\n\n'
            '-----------------------------------------------\n'
            '------------------  Training ------------------\n'
            '-----------------------------------------------\n'
            'Training {} Starting epoch {}/{}.'.format(self.tag, self.current_epoch, self.total_epoches)
        )
        time.sleep(0.5)
        for module in self.model.values():
            module.train()

        train_loss = 0.
        torch.cuda.empty_cache()
        y_trained_true = []
        y_trained_pred = []
        for index, data in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            loss, pred, true = self.batrain(index, data)
            train_loss += loss
            y_trained_pred.append(pred)
            y_trained_true.append(true)

        self.metrics(y_pred=y_trained_pred, y_true=y_trained_true, isTrain=True)

        if self.use_tboard:
            self.tboard_writer.add_scalar('loss', train_loss, global_step=self.current_epoch)

        self.show_outputs('\n{} Epoch{} finished ! Loss: {}\n'.format(self.tag, self.current_epoch, train_loss))

    ## 进行预测，需要重构
    def predict(self, data):
        '''
        验证过程中预测
        :param data: [for index, data in tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader))]
        :return: pred, labels [lists]
        '''
        raise NotImplementedError

    ## 验证器进行验证
    def valider(self):
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '-----------------  Evaluation  ----------------\n'
            '-----------------------------------------------'
        )
        time.sleep(0.5)

        for module in self.model.values():
            module.eval()

        y_true = []
        y_pred = []
        for index, data in tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader)):
            inputs, labels = data
            inputs = inputs.float()
            labels = labels.float()

            pred = self.predict(inputs)
            true = labels.numpy()

            y_pred.append(pred)
            y_true.append(true)

        index = self.metrics(y_pred=y_pred, y_true=y_true)
        self.monitor.append(index)
        if index > self.best_index:
            self.best_index = index
            self.best_epoch = self.current_epoch

    ## 指标，越大越好 (对于越小越好的指标进行取反)
    def metrics(self, y_pred, y_true, isTrain=False, num_cal_metric=-1):
        '''
        判断训练集 、 验证集的指标
        并将监控量进行赋值
        :param y_pred: list
        :param y_true: list
        :param num_cal_metric: 计算metric的抽检样本个数 （一般用于训练），-1:全部计算
        :return moniter that is use to adapt lr （注意这里必须是越大越好的，如果不是进行取反）
        '''

        raise NotImplementedError

    ## 更新器进行训练参数更新
    def updater(self):
        ## 学习率下降
        for key, (type, scheduler) in self.scheduler.items():
            if type == 'ReduceLROnPlateau':
                scheduler.step(self.monitor[-1])
            elif type == 'CosineAnnealingLR':
                scheduler.step()
            elif type == 'CustomizeLR':
                scheduler.step()
            else:
                assert 'Unknown Scheduler!'

            self.show_outputs(
                "current learning rate: {}".format(
                    self.optimizer[key].state_dict()['param_groups'][0]['lr']))

    ## 保存器保存当前Trainer过程
    def save_er(self):
        if self.save_mode == 'key_epoch':
            if self.best_epoch == self.current_epoch:
                self.save_trainer(path=os.path.join(self.cpnt_dire, 'Tag-{}-best-checkpoint.pth'.format(self.tag)))
            self.save_trainer(path=os.path.join(self.cpnt_dire, 'Tag-{}-latest-checkpoint.pth'.format(self.tag)))

        elif self.save_mode == 'all_epoch':
            time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            path = os.path.join(self.cpnt_dire, "Tag-{}-Epoch-{}-cp-{}.pth".format(
                self.tag, self.current_epoch, time_stamp))
            self.save_trainer(path=path)

    ## 进行训练
    def run(self, DEBUG=False):
        ## 初始化训练器
        self.init_trainer()

        ## 开始训练
        self.show_outputs(
            '\n'
            '-----------------------------------------------\n'
            '-----------------  Run Trainer  ---------------\n'
            '-----------------------------------------------\n'
            'Total epoches:{}\n'
            'Current epoch:{}\n'
            'Criterion: {}\n'
                .format(self.total_epoches, self.current_epoch, self.criterions)
        )
        current_epoch = self.current_epoch

        ## 这里需要注意的是，这里最好的epoch/index是不继承自保存的.pth中的，所以是从新开始的，cur_epoch需要在.yaml参数中设置
        self.monitor = []  ## 指标监视器
        self.best_epoch = -1  ## 当前结果最好的epoch
        self.best_index = -1  ## 最好结果的指标
        self.valider()
        self.save_trainer(path=os.path.join(self.cpnt_dire, 'Tag-{}-Initial.pth'.format(self.tag)))
        for epoch in range(current_epoch, self.total_epoches):
            self.current_epoch = epoch
            if DEBUG:
                self.trainer()
                if epoch % self.validation_step == 0:
                    self.valider()
                    self.save_er()
                    self.updater()
            else:
                try:
                    self.trainer()
                    if epoch % self.validation_step == 0:
                        self.valider()
                        self.save_er()
                        self.updater()
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc(file=open('./error.log', 'w+'))
                    self.save_trainer(path=os.path.join(self.cpnt_dire, 'Tag-{}-ERROR.pth'.format(self.tag)))
                    break

    ## 展示输出
    def show_outputs(self, info):
        if hasattr(self, 'use_logger') and self.use_logger: logging.info(info)
        if not (hasattr(self, 'use_printt') and not self.use_pprint): print(info)


## Segmentation Trainer Base
class SegTrainerBase(TrainerBase):
    def forward(self, index, data):
        images, labels = data
        images = images.float()
        labels = labels.float()

        if self.use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        predict, _ = self.model['Unet'](images)

        return predict, labels

    def predict(self, data):
        images = data
        if self.use_cuda:
            images = images.cuda()

        with torch.no_grad():
            predict, _ = self.model['Unet'](images)

        if self.use_cuda:
            predict = predict.cpu()

        return predict

    def metrics(self, y_pred, y_true, isTrain=False, num_cal_metric=-1):

        if num_cal_metric > 0:
            idx = np.random.randint(0, len(y_true), num_cal_metric)
            y_true = [y_true[i] for i in idx]
            y_pred = [y_pred[i] for i in idx]

        dices = []
        for pred, true in tqdm(zip(y_pred, y_true)):
            dice = get_dice(np.asarray(pred > 0.5).astype(int), true)
            dices.append(dice)

        self.show_outputs("\nVal num:{}\ndice:{}\n".format(len(dices), np.mean(dices)))
        if hasattr(self, 'use_tboard') and self.use_tboard:
            self.tboard_writer.add_scalar(
                '{} Dice'.format('Train' if isTrain else 'Valid'), np.mean(dices), global_step=self.current_epoch)

        return np.mean(dices)


## Classification Trainer Base
class ClsTrainerBase(TrainerBase):
    def forward(self, index, data):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.long()

        if self.use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        feature = self.model['extractors'](inputs)
        predict = self.model['classifier'](feature)
        return predict, labels

    def predict(self, data):
        images = data
        if self.use_cuda:
            images = images.cuda()

        with torch.no_grad():
            feature = self.model['extractors'](images)
            predict = self.model['classifier'](feature)

        if self.use_cuda:
            predict = predict.cpu()

        predict = list(map(int, list(predict.numpy().argmax(axis=1))))
        return np.asarray(predict)

    def metrics(self, y_pred, y_true, isTrain=False, num_cal_metric=-1):
        y_pred = np.concatenate([pred.argmax(axis=1) for pred in y_pred], axis=0)
        y_true = np.concatenate(y_true, axis=0)

        y_pred = [int(i) for i in y_pred]
        y_true = [int(i) for i in y_true]

        many_shot, median_shot, feww_shot = shot_acc(list(y_pred), list(y_true), self.train_dataloader)
        self.class_report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
        matrix = confusion_matrix(y_true, y_pred)

        self.show_outputs(
            "\nVal num:{}\nreport:\n{}\nPrecision:{}\nRecall:{}\nF1:{}\nAccuracy:{}\nAccuracy:{} \nshot acc: [ many:{} median:{} few:{} ] \nepoch:{}\nbest epoch:{}\nbest accuracy:{}\n".format(
                len(y_true),
                # y_pred,
                # y_true,
                classification_report(y_true=y_true, y_pred=y_pred),
                get_multi_precision(y_pred, y_true),
                get_multi_recall(y_pred, y_true),
                get_multi_f1(y_pred, y_true),
                matrix.diagonal() / matrix.sum(axis=1),
                get_multi_accuracy(y_pred, y_true),
                many_shot, median_shot, feww_shot,
                self.current_epoch if hasattr(self, 'current_epoch') else 'None',
                self.best_epoch,
                self.best_index
            ))

        if hasattr(self, 'use_tboard') and self.use_tboard:
            self.tboard_writer.add_scalar(
                '{} Accuracy'.format('Train' if isTrain else 'Valid'), get_multi_accuracy(y_pred, y_true),
                global_step=self.current_epoch)
        return get_multi_accuracy(y_pred=y_pred, y_true=y_true)


def train():
    '''
    注意：
    1. 在Windows测试时,num_worker不可用，所以需要设置为0,否则dataloader过不去
    '''
    from mtools.mio import get_yaml
    trainer = SegTrainerBase(config=get_yaml('./mtools/mtorch/config-seg-train.yaml'))
    trainer = ClsTrainerBase(config=get_yaml('./mtools/mtorch/config-cls-train.yaml'))

    trainer.run()
    # trainer.save_trainer(path='./ga.pth')

    # trainer.run(DEBUG=True)


if __name__ == '__main__':
    train()
