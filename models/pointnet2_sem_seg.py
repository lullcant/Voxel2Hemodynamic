import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        # self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 3 + 3, [32, 32, 64], False)
        # self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        # self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        # self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)

        # self.sa1 = PointNetSetAbstraction(2048, 0.1, 32, 5 + 3, [32, 32, 64], False)
        # self.sa2 = PointNetSetAbstraction(1024, 0.15, 32, 64 + 3, [64, 64, 128], False)
        # self.sa3 = PointNetSetAbstraction(512, 0.2, 32, 128 + 3, [128, 128, 256], False)
        # self.sa4 = PointNetSetAbstraction(256, 0.3, 32, 256 + 3, [256, 256, 512], False)

        self.sa1 = PointNetSetAbstraction(2048, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(1024, 0.15, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(256, 0.2, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(64, 0.4, 32, 256 + 3, [256, 256, 512], False)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        exit()

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = self.sigmoid(x)  # ffr
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward_ffr(self, pred, target, trans_feat, weight, epoch):
        # total_loss = F.nll_loss(pred, target, weight=weight)
        diff_weight = 5 # 5
        
        # only top-branch (30%) will be used to calculate loss
        pred = pred.view(2, -1, 1)
        pred = pred[:, :1360].reshape(-1, 1)
        target = target.view(2, -1, 1)
        target = target[:, :1360].reshape(-1, 1)
        
        # FFR < 0.6 will not be trained
        mask = torch.zeros(target.shape, device=target.device)
        mask[target>=0.5] = 1
        pred_diff = torch.abs(pred - target)
        pred_diff = pred_diff[torch.nonzero(mask, as_tuple=True)]
        if pred_diff.shape[0] == 0:
            print(f"************* Impossible...")
            loss = torch.tensor(0.001, device=target.device, requires_grad=True)
            return loss

        # pred_diff = pred_diff.reshape(-1, 1)
        total_loss = (diff_weight*(torch.mean(pred_diff)))**2
        return total_loss

    def forward(self, pred, target, trans_feat, weight, epoch):
        # total_loss = F.nll_loss(pred, target, weight=weight)

        # get_min = lambda x:torch.min(x, dim=1, keepdim=True)[0]
        # get_max = lambda x:torch.max(x, dim=1, keepdim=True)[0]

        # batch_size = 2
        # num_class = pred.shape[-1]
        # pred = pred.view(batch_size, -1, num_class)
        # # pred = (pred - get_min(pred)) / (get_max(pred) - get_min(pred))
        # pred = pred[:, :4096].reshape(-1, num_class)
        # target = target.view(batch_size, -1, num_class)
        # # target = (target - get_min(target)) / (get_max(target) - get_min(target))
        # target = target[:, :4096].reshape(-1, num_class)

        weight = 1
        # loss = weight * torch.mean((pred - target)**2)
        loss = weight * self.loss(pred, target)
        return loss

if __name__ == '__main__':
    import  torch
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))