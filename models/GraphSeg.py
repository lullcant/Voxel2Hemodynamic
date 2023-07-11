import torch
import trimesh
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential, GCNConv
from mtools.mtorch.models.graph.GResNet import GBottleNeck
from mtools.mtorch.models.graph.GUnpooling import GUnpooling
from mtools.mtorch.models.graph.GProjection import GProjection3D


class Sphere(object):
    '''
    load three spheres (interpolation)
    '''

    def __init__(self):
        '''
        Initialize the sphere
        verts [num_verts, 3]
        edges [2, num_edges] : [torch.Size([2, 480]), torch.Size([2, 1920]), torch.Size([2, 7680])]
        faces [num_faces, 3] : [torch.Size([320, 3]), torch.Size([1280, 3]), torch.Size([5120, 3])]
        '''
        self.spheres = [trimesh.load('./utils/sphere-{}.obj'.format(i)) for i in range(3)]
        self.verts = [torch.from_numpy(np.asarray(mesh.vertices).copy()).float() for mesh in self.spheres]
        self.edges = [torch.from_numpy(np.asarray(mesh.edges_unique).copy()).long().transpose(1, 0) for mesh in
                      self.spheres]
        self.faces = [torch.from_numpy(np.asarray(mesh.faces).copy()).long() for mesh in self.spheres]


class GraphSeg(nn.Module):
    def __init__(self, coords_dim=3, hidden_dim=192, num_blcoks=6, feats_dims=[128, 64, 32]):
        super(GraphSeg, self).__init__()

        self.init_sphere = Sphere()

        ## network parameter

        ## network
        self.gcn_projn = nn.ModuleList([
            GProjection3D(num_feats) for num_feats in feats_dims
        ])
        self.gcn_model = nn.ModuleList([
            GBottleNeck(block_num=num_blcoks, in_dim=sum(feats_dims) + coords_dim,
                        hidden_dim=hidden_dim, out_dim=coords_dim),
            GBottleNeck(block_num=num_blcoks, in_dim=sum(feats_dims) + hidden_dim + coords_dim,
                        hidden_dim=hidden_dim, out_dim=coords_dim),
            GBottleNeck(block_num=num_blcoks, in_dim=sum(feats_dims) + hidden_dim + coords_dim,
                        hidden_dim=hidden_dim, out_dim=hidden_dim),
        ])
        self.gcn_final = Sequential(
            'x, edge_index', [
                (nn.ReLU(), 'x -> x'),
                (GCNConv(in_channels=hidden_dim, out_channels=coords_dim), 'x, edge_index -> x')
            ]
        )
        self.gcn_unpol = GUnpooling()

    def forward(self, img_feats):
        '''
        :param img_feats: [torch.Size([batch, 128, 8, 8,  8 ]),
                           torch.Size([batch, 64, 16, 16, 16]),
                           torch.Size([batch, 32, 32, 32, 32])]
        :return: verts  x1-[batch, 162, 3]  x2-[batch, 642,  3]  x3-[batch, 2562, 3]
                 faces     [batch, 320, 3]     [batch, 1280, 3]     [batch, 5120, 3]
        '''

        batch = img_feats[0].size()[0]
        device = img_feats[0].device

        ## Initilization
        ## init_verts [batch, 162, 3]
        ## init_edges [torch.Size([2, 480]), torch.Size([2, 1920]), torch.Size([2, 7680])]
        init_verts = self.init_sphere.verts[0].unsqueeze(dim=0).expand(batch, -1, -1).to(device)
        init_edges = [e.to(device) for e in self.init_sphere.edges]
        init_faces = [f.unsqueeze(dim=0).expand(batch, -1, -1).to(device) for f in self.init_sphere.faces]

        ## GCN Block #1
        ## x1_proj   [batch, 162, num_feats]
        ## x1        [batch, 162，coord_dim]
        ## x_hidden  [batch, 162, hidden_dim]
        x1_proj = torch.cat([self.gcn_projn[idx](images=feats, verts=init_verts)
                             for idx, feats in enumerate(img_feats)], dim=2)
        x1, x_hidden = self.gcn_model[0](x=torch.cat([x1_proj, init_verts], dim=2), edge_index=init_edges[0])
        x1 = x1 + init_verts

        # ===============================================================

        ## GCN Block #2
        ## x1_unpool        [batch, 642, coord_dim]
        ## x1_hidden_unpool [batch, 642, hidden_dim]
        ##
        ## x2_proj          [batch, 642, num_feats]
        ## x2               [batch, 642, coord_dim]
        ## x_hidden         [batch, 642, hidden_dim]
        x1_unpool = self.gcn_unpol(x=x1, edge_index=init_edges[0])
        x1_hidden_unpool = self.gcn_unpol(x=x_hidden, edge_index=init_edges[0])

        x2_proj = torch.cat([self.gcn_projn[idx](images=feats, verts=x1_unpool)
                             for idx, feats in enumerate(img_feats)], dim=2)
        x2, x_hidden = self.gcn_model[1](
            x=torch.cat([x1_hidden_unpool, x2_proj, x1_unpool], dim=2), edge_index=init_edges[1])
        x2 = x2 + x1_unpool

        # ===============================================================

        ## GCN Block #3
        ## x2_unpool        [batch, 2562, coord_dim]
        ## x2_hidden_unpool [batch, 2562, hidden_num]
        ##
        ## x3_proj          [batch, 2562, num_feats]
        ## x3               [batch, 2562, hidden_num]
        x2_unpool = self.gcn_unpol(x=x2, edge_index=init_edges[1])
        x2_hidden_unpool = self.gcn_unpol(x=x_hidden, edge_index=init_edges[1])

        x3_proj = torch.cat([self.gcn_projn[idx](images=feats, verts=x2_unpool)
                             for idx, feats in enumerate(img_feats)], dim=2)
        x3, _ = self.gcn_model[2](
            x=torch.cat([x2_hidden_unpool, x3_proj, x2_unpool], dim=2), edge_index=init_edges[2])

        # ===============================================================

        ## GCN Block FINAL
        x3 = self.gcn_final(x3, init_edges[2])
        x3 = x3 + x2_unpool

        return [x1, x2, x3], init_faces


def create_model(coords_dim=3, hidden_dim=192, feats_dims=[128, 64, 32], *args):
    gcn = GraphSeg(coords_dim=coords_dim, hidden_dim=hidden_dim, feats_dims=feats_dims)
    return gcn


if __name__ == '__main__':
    from mtools.mtorch.models.Unet3D import Unet3D

    images = torch.rand(size=[2, 1, 32, 32, 32])
    unet = Unet3D(in_channels=1, out_channels=1, channels=[64, 128, 256])
    gseg = GraphSeg(feats_dims=[128, 64, 32])

    predict, skips = unet(images)
    points, faces = gseg(skips)

    print([i.size() for i in points])
    print([i.size() for i in faces])
