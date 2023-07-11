import torch
import torch.nn as nn
from mtools.mtorch.loss.MeshLoss import MeshLoss


class GraphLoss(nn.Module):

    def __init__(self, weight_chamfer=1.0, weight_edge=0.1, weight_norm=0.01, weight_lapa=0.1):
        super(GraphLoss, self).__init__()
        self.criterion = MeshLoss(weight_chamfer, weight_edge, weight_norm, weight_lapa)

    def forward(self, inputs, target, *args):
        '''
        :param inputs:
                - recon  [batch, 1, 32, 32, 32]
                - verts  x1-[batch, 162, 3]  x2-[batch, 642, 3]  x3-[batch, 2562, 3]
                - faces            [320, 3]           [1280, 3]            [5120, 3]
        :param target:
        :param args:
        :return:
        '''
        _, p_verts, p_faces = inputs
        _, verts = target

        chamfer_loss, edge_loss, norm_loss, lapa_loss = 0., 0., 0., 0.
        for vert, face in zip(p_verts, p_faces):
            chamfer, lapa, edge, norm = self.criterion([vert, face], verts)

            chamfer_loss += chamfer
            edge_loss += edge
            norm_loss += norm
            lapa_loss += lapa

        loss = chamfer_loss + lapa_loss + \
               edge_loss + norm_loss

        return loss


def create_loss(weight=None, reduction='mean'):
    return GraphLoss()
