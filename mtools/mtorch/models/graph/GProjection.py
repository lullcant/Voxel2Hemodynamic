import torch
import torch.nn as nn
import torch.nn.functional as F


class GProjection3D(nn.Module):

    def __init__(self, num_feats):
        super(GProjection3D, self).__init__()

        self.sum_neighbourhood = nn.Conv2d(num_feats, num_feats, kernel_size=(1, 27), padding=0)

        # torch.nn.init.kaiming_normal_(self.sum_neighbourhood.weight, nonlinearity='relu')
        # torch.nn.init.constant_(self.sum_neighbourhood.bias, 0)
        self.shift_delta = nn.Conv1d(num_feats, 27 * 3, kernel_size=(1), padding=0)
        self.shift_delta.weight.data.fill_(0.0)
        self.shift_delta.bias.data.fill_(0.0)

        self.feature_diff_1 = nn.Linear(num_feats + 3, num_feats)
        self.feature_diff_2 = nn.Linear(num_feats, num_feats)

        self.feature_center_1 = nn.Linear(num_feats + 3, num_feats)
        self.feature_center_2 = nn.Linear(num_feats, num_feats)

    def forward(self, images, verts):
        B, N, _ = verts.shape

        w, h, d = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2]

        w = torch.clamp(w, min=-1, max=1)
        h = torch.clamp(h, min=-1, max=1)
        d = torch.clamp(d, min=-1, max=1)

        verts = torch.stack([w, h, d], dim=-1)


        center = verts[:, :, None, None]
        features = F.grid_sample(images, center, mode='bilinear', padding_mode='border', align_corners=True)
        features = features[:, :, :, 0, 0]
        # setting first shift to zero so it samples at the exact point
        shift_delta = self.shift_delta(features).permute(0, 2, 1).view(B, N, 27, 1, 3)
        shift_delta[:, :, 0, :, :] = shift_delta[:, :, 0, :, :] * 0

        # neighbourhood = vertices[:, :, None, None] + self.shift[:, :, :, None] + shift_delta
        neighbourhood = verts[:, :, None, None] + shift_delta
        features = F.grid_sample(images, neighbourhood, mode='bilinear', padding_mode='border',
                                 align_corners=True)
        features = features[:, :, :, :, 0]
        features = torch.cat([features, neighbourhood.permute(0, 4, 1, 2, 3)[:, :, :, :, 0]], dim=1)

        features_diff_from_center = features - features[:, :, :, 0][:, :, :,
                                               None]  # 0 is the index of the center cordinate in shifts
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])
        features_diff_from_center = self.feature_diff_1(features_diff_from_center)
        features_diff_from_center = self.feature_diff_2(features_diff_from_center)
        features_diff_from_center = features_diff_from_center.permute([0, 3, 2, 1])

        features_diff_from_center = self.sum_neighbourhood(features_diff_from_center)[:, :, :, 0].transpose(2, 1)

        center_feautres = features[:, :, :, 13].transpose(2, 1)
        center_feautres = self.feature_center_1(center_feautres)
        center_feautres = self.feature_center_2(center_feautres)

        features = center_feautres + features_diff_from_center
        return features
