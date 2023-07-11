## DDP并行模式下的训练器
from mtool.mtorch.mtrainer import *
import random
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from contextlib import contextmanager


@contextmanager
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    # 这里的用法其实就是协程的一种哦。
    yield
    if rank == 0:
        torch.distributed.barrier()


'''
运行方式：
 
- 单机多卡
  python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 train.py
  - torch.distributed.launch: 将local_rank注入到进程中
  - nproc_per_node: 表示每个节点需要创建多少个进程(使用几个GPU就创建几个)；
  - nnodes: 表示使用几个节点，因为我们是做单机多核训练，所以设为1。
  
- 多机多卡

'''


class DDPTBase(TBase):
    def __init__(self, model, rank=None, word_size=-1):
        super(DDPTBase, self).__init__(model=model)
        self.param_rank = rank
        self.word_size = word_size

    ## 初始化日志
    def init_loggers(self):

        LOG_FORMAT = "%(asctime)s - %(levelname)s: %(message)s"

        ## 创建一个 logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  ## logger等级总开关

        ## 创建一个写入日志文件的handler
        path = './{}.log'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
        file_handler = logging.FileHandler(path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  ## 输出到文件的log等级
        file_handler.setFormatter(LOG_FORMAT)  ## 输出到文件的log格式

        ## 创建一个写到console的handler
        cons_handler = logging.StreamHandler()
        cons_handler.setLevel(logging.DEBUG if dist.get_rank() == 0 else logging.WARN)
        cons_handler.setFormatter(LOG_FORMAT)

        ## 将logger添加到handler里面
        logger.addHandler(cons_handler)
        logger.addHandler(file_handler)

    def init_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_ddp(self):
        ## 进程通过local_rank来标识自己
        # local_rank = 0 为主进程; 其他为slave进程
        '''
        torch.distributed.launch 以命令行参数的方式将args.local_rank变量
        注入到每个进程中每个进程得到的变量值都不相同。比如使用 4 个GPU的话，
        则 4 个进程获得的args.local_rank值分别为0、1、2、3。
        '''
        if self.param_rank is None:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--local_rank", type=int, default=-1)
            args = parser.parse_args()
            self.local_rank = args.local_rank
        else:
            self.local_rank = self.param_rank

        ## 分配后端，nccl是GPU设备上最快、最推荐的后端，用于进程间通信
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)
        dist.init_process_group(backend='nccl', rank=self.local_rank, world_size=self.word_size)

        ## 宇宙的秘密
        self.init_seed(42 + dist.get_rank())

    ## 初始化数据集
    def init_dataset(self, train_dataset, valid_dataset, batch_size, num_worker):
        '''
        :param train_dataset: 训练数据集
        :param valid_dataset: 验证数据集
        :param batch_size:
        :param num_worker:
        :return:
        '''

        train_nums = train_dataset.__len__()
        valid_nums = valid_dataset.__len__()

        logging.info(
            '\n'
            '-----------------------------------------------\n'
            '----------------  Data Loading ----------------\n'
            '-----------------------------------------------\n'
            'Sing device Batch  size:{}\n'
            'Worker num:{}\n'
            'Train  num:{}\n'
            "Valid  num:{}\n"
                .format(batch_size, num_worker, train_nums, valid_nums)
        )

        ## sampler 与 shuffle 冲突
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker,
                                           sampler=train_sampler, drop_last=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=False)

    def init_network(self, curr_params=None, init_method='kaiming', use_cuda=False, use_parallel=False):
        '''
        :param curr_params: 当前的神经网络参数
        :param init_method: 初始化神经网络方法
        :return:
        '''

        logging.info(
            '\n'
            '-----------------------------------------------\n'
            '----------------  Model Loading ---------------\n'
            '-----------------------------------------------\n'
            'Model Structure:\n{}\n'
            'Current param: {}\n'
            'Init   method: {}\n'
            'Use CUDA:      {}\n'
            "Use Parallel:  {}\n"
                .format(self.model, curr_params, init_method, use_cuda, use_parallel)
        )

        ## 加载模型参数 / 初始化模型
        if curr_params is not None and dist.get_rank() == 0:
            self.load_param(checkpoint=curr_params)
        # 对网络进行初始化，选择初始化方法
        elif init_method is not None and dist.get_rank() == 0:
            self.init_weights(method=init_method)

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

    def run(self, datasets, criterion, config, DEBUG=False):
        '''
        :param datasets: dicts:{'train':train_dataset;'valid':valid_dataset}
        :param criterion:
        :return:
        '''
        self.config = config

        ## 初始化DDP
        self.use_cuda = self.config['network']['use_cuda']
        self.use_parallel = self.config['network']['use_parallel']
        assert torch.cuda.is_available() and self.use_cuda and self.use_parallel, "CUDA is not supported!"
        self.init_ddp()

        ## 初始化日志
        ## 初始化日志
        self.use_logger = self.config['monitor']['logger']
        self.use_printt = self.config['monitor']['stdstream']
        self.use_tboard = self.config['monitor']['tensorboardx']

        if self.use_logger: self.init_loggers()
        if self.use_tboard: self.init_tfboard()
        self.tag = self.config['tag']

        ## 初始化数据集
        self.init_dataset(
            train_dataset=datasets['train'], valid_dataset=datasets['valid'],
            batch_size=self.config['dataset']['batch_size'],
            num_worker=self.config['dataset']['num_worker']
        )

        ## 初始化神经网络
        self.init_network(
            curr_params=self.config['network']['curr_params'],
            init_method=self.config['network']['init_method'],
            use_cuda=self.use_cuda,
            use_parallel=self.use_parallel
        )

        ## 初始化优化器和策略器
        self.init_optimer(self.config['optimizer'], self.config['scheduler'])

        ## 初始化损失函数
        self.criterion = criterion

        ## 开始训练
        self.dire_checkpoint = self.config['saver']['dire_checkpoint']
        self.thresh_gradnt = self.config['train']['threshold_grad']  ## 梯度裁剪
        self.total_epoches = self.config['train']['total_epoches']
        current_epoch = self.config['train']['current_epoch']
        logging.info(
            '\n'
            '-----------------------------------------------\n'
            '-----------------  Run Trainer  ---------------\n'
            '-----------------------------------------------\n'
            'Total epoches:{}\n'
            'Current epoch:{}\n'
            'Criterion: {}\n'
                .format(self.total_epoches, current_epoch,
                        self.criterion)
        )

        self.current_epoch = current_epoch
        if dist.get_rank() == 0:
            self.valider()
        for epoch in range(current_epoch, self.total_epoches):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(self.current_epoch)

            if DEBUG:
                self.trainer()
                if dist.get_rank() == 0:
                    self.save_param()
                    self.valider()
                    self.updater()
            else:
                try:
                    self.trainer()
                    if dist.get_rank() == 0:
                        self.save_param()
                        self.valider()
                        self.updater()
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc(file=open('./error.log', 'w+'))
                    if dist.get_rank() == 0:
                        self.save_param(tag='error')
                    break


def run_demo(ddptest, world_size):
    from torch.multiprocessing.spawn import spawn
    spawn(
        ddptest,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


def DDPtest():
    config = {
        'tag': 'LT-classifier',
        'dataset': {
            'batch_size': 20,
            'num_worker': 0
        },
        'network': {
            'curr_params': None,
            'init_method': 'kaiming',
            'use_cuda': True,
            'use_parallel': True
        },
        'optimizer': {
            'Adam': {'lr': 0.01}
        },
        'scheduler': {
            'ReduceLROnPlateau': {
                'mode': 'max',
                'factor': 0.1,
                'patience': 1}
        },
        'train': {
            'current_epoch': 1,
            'total_epoches': 1000,
            'threshold_grad': 10000.0
        },
        'valid': {'step': 1},
        'saver': {'dire_checkpoint': './checkpoints'}
    }

    import torchvision

    model = torchvision.models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=10)
    trainer = DDPTBase(model)

    criterion = nn.CrossEntropyLoss()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    valid_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainer.run(criterion=criterion, datasets={'train': train_dataset, 'valid': valid_dataset}, config=config,
                DEBUG=True)


if __name__ == '__main__':
    DDPtest()
