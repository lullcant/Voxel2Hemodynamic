import torch
import torch.nn as nn
import numpy as np


class GUnpooling(nn.Module):
    '''
    Unpooling verts in graph
    '''
    def __init__(self):
        super(GUnpooling, self).__init__()

    def forward(self, x, edge_index):
        '''
        :param x: [batch, num_verts, dim]
        :param edge_index: [2, num_edges]
        :return: [batch, num_verts + num_edges, dim]
        '''

        new_features = x[:, edge_index.transpose(1, 0)].clone()
        new_vertices = 0.5 * new_features.sum(2)
        output = torch.cat([x, new_vertices], 1)

        return output


if __name__ == '__main__':
    import trimesh

    # faces = trimesh.load('../../../../utils/sphere-1.obj').faces
    verts = torch.tensor(trimesh.load('../../../../utils/sphere-1.obj').vertices, dtype=torch.float)[np.newaxis, :, :]
    edges = torch.tensor(trimesh.load('../../../../utils/sphere-1.obj').edges_unique, dtype=torch.long).transpose(1, 0)

    faces = trimesh.load('../../../../utils/sphere-2.obj').faces

    model = GUnpooling()
    x = model(verts, edges)

    print(x.size())


    exit()

    trimesh.Trimesh(x, faces).export('../../../../s2ga.obj')
