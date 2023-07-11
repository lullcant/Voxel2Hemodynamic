from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
import torch.nn as nn
import numpy as np
import trimesh
import torch


class MeshLoss(nn.Module):
    def __init__(self, weight_chamfer=1.0, num_samples=5000, weight_edge=1.0, weight_norm=0.1, weight_lapa=0.1):
        super(MeshLoss, self).__init__()
        self.weight_chamfer = weight_chamfer
        self.weight_edge = weight_edge
        self.weight_norm = weight_norm
        self.weight_lapa = weight_lapa

    def forward(self, input, target, *args):
        '''
        :param inputs: [verts, faces]
                    - in_verts (batch, num_point_{p}, 3)
                    - faces    (batch, num_faces_{p}, 3)
        :param target: ground truth verts (num_point_{t}, 3)
        :return:
        '''
        in_verts, faces = input
        gt_verts = target

        mesh = Meshes(verts=in_verts, faces=faces)

        chamfer = chamfer_distance(sample_points_from_meshes(mesh,num_samples=gt_verts.size()[1]), gt_verts)[0]
        # chamfer = chamfer_distance(mesh.verts_padded(), gt_verts)[0]
        chamfer = chamfer * self.weight_chamfer

        lapa = mesh_laplacian_smoothing(mesh) * self.weight_lapa
        edge = mesh_edge_loss(mesh) * self.weight_edge
        norm = mesh_normal_consistency(mesh) * self.weight_norm
        return chamfer, lapa, edge, norm


if __name__ == '__main__':
    mesh = trimesh.load('../../../utils/sphere-0.obj')

    verts = torch.tensor(mesh.vertices).unsqueeze(dim=0).expand(2, -1, -1).float()
    faces = torch.tensor(mesh.faces).unsqueeze(dim=0).expand(2, -1, -1).long()

    mesh = trimesh.load('../../../utils/sphere-2.obj')
    nverts = torch.tensor(mesh.vertices).unsqueeze(dim=0).expand(2, -1, -1).float()

    print(verts.size())
    print(faces.size())
    print(nverts.size())

    criterion = MeshLoss()
    print(criterion([verts, faces], nverts))

    # print(verts.size())
    # print(faces.size())
