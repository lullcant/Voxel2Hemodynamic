from torch_geometric.nn import Sequential, GCNConv
from pytorch3d.structures.meshes import Meshes
import torch.nn.functional as F
import SimpleITK as sitk
import torch.nn as nn
import numpy as np
import trimesh
import torch


class GResBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GResBlock, self).__init__()
        self.model = Sequential('x, edge_index', [
            (GCNConv(in_channels=in_dim, out_channels=hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            (GCNConv(in_channels=hidden_dim, out_channels=in_dim), 'x, edge_index -> x'),
            nn.ReLU()
        ])

    def forward(self, x, edge_index):
        '''
        :param x: [batch, num_verts, channel]
        :param edge_index: [2, num_edges]
        :return:
        '''
        return (x + self.model(x, edge_index)) * 0.5


class GBottleNeck(nn.Module):
    def __init__(self, block_num, in_dim, hidden_dim, out_dim):
        super(GBottleNeck, self).__init__()

        self.model = Sequential('x, edge_index', [
            (GCNConv(in_channels=in_dim, out_channels=hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(),
            *[(GResBlock(in_dim=hidden_dim, hidden_dim=hidden_dim), 'x, edge_index -> x') for _ in range(block_num)]
        ])
        self.gconv = GCNConv(in_channels=hidden_dim, out_channels=out_dim)

    def forward(self, x, edge_index):
        '''
        :param x: [batch, num_verts, channel]
        :param edge_index: [2, num_edges]
        :return:
        '''
        x_hidden = self.model(x=x, edge_index=edge_index)
        return self.gconv(x_hidden, edge_index), x_hidden


if __name__ == '__main__':
    in_dim, hidden_dim = 3, 12
    x = torch.rand(size=[3, 8, 3])
    y = torch.randint(low=0, high=8, size=[2, 8])

    model = GBottleNeck(block_num=6, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=4)

    x, x_hidden = model(x=x, edge_index=y)
    print(x)
    print(x.size())
    print(x_hidden.size())
