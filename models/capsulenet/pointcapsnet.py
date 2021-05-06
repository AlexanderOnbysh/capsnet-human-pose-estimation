from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()
        # (2048,6)
        # TODO: upadte accoring to number of channels
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # Conv1D
        # self.conv1 = torch.nn.Conv1d(6, 64, 1)  # Conv1D
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)  # Norm
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.size())     (4,128,2048)
        return x


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=8, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        # 共16结构
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),  # (4,1024,2048)
                ('bn3', nn.BatchNorm1d(1024)),  # Norm
                ('mp1', torch.nn.MaxPool1d(num_points)),  # (4,1024,1)
            ]))
            for _ in range(prim_vec_size)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        # print("u[0].size" , u[0].size())
        # (4,1024,1)
        u = torch.stack(u, dim=2)
        # print(u.size())
        return self.squash(u.squeeze())

    # activation
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        # print("square",squared_norm.size())
        # (4,1024,1)
        output_tensor = squared_norm * input_tensor / \
                        ((1. + squared_norm) * torch.sqrt(squared_norm))
        if (output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor


class LatentCapsLayer(nn.Module):
    def __init__(self, latent_caps_size=16, prim_caps_size=1024, prim_vec_size=16, latent_vec_size=64):
        super(LatentCapsLayer, self).__init__()
        self.prim_vec_size = prim_vec_size
        self.prim_caps_size = prim_caps_size
        self.latent_caps_size = latent_caps_size
        self.W = nn.Parameter(0.01 * torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size))
        # self.W.requires_grad=False
        # self.W = 0.01*torch.randn(latent_caps_size, prim_caps_size, latent_vec_size, prim_vec_size).cuda()
        # self.W.requires_grad_(True)

    def forward(self, x):
        u_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        # print(u_hat.requires_grad)  True
        u_hat_detached = u_hat.detach()
        # print(u_hat_detached.requires_grad) false
        b_ij = Variable(torch.zeros(x.size(0), self.latent_caps_size, self.prim_caps_size)).cuda()
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, 1)
            if iteration == num_iterations - 1:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat, dim=-2, keepdim=True))
            else:
                v_j = self.squash(torch.sum(c_ij[:, :, :, None] * u_hat_detached, dim=-2, keepdim=True))
                b_ij = b_ij + torch.sum(v_j * u_hat_detached, dim=-1)
        return v_j.squeeze(-2)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
                        ((1. + squared_norm) * torch.sqrt(squared_norm))
        # print(output_tensor.size())  (4,32,1,16)
        return output_tensor


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size / 2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size / 2), int(self.bottleneck_size / 4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size / 4), 3, 1)

        self.th = torch.nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 4))

    def forward(self, x):
        # print(x.size())      (4,18,32)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("1",x.size())   (4,18,32)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("2",x.size())   (4,9,32)
        x = F.relu(self.bn3(self.conv3(x)))
        # print("3",x.size())   (4,4,32)
        x = self.th(self.conv4(x))
        # print("4",x.size())   (4,3,32)
        return x


class CapsDecoder(nn.Module):
    def __init__(self, latent_caps_size, latent_vec_size, num_points):
        super(CapsDecoder, self).__init__()
        self.latent_caps_size = latent_caps_size  # 32
        self.bottleneck_size = latent_vec_size  # 16
        self.num_points = num_points
        self.nb_primitives = int(num_points / latent_caps_size)  # 2048/32 = 64
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=self.bottleneck_size + 2) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        # print(x.size()) (8,32,16)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.latent_caps_size))
            # print(rand_grid.size()) #(8,2,32)
            rand_grid.data.uniform_(0, 1)
            y = torch.cat((rand_grid, x.transpose(2, 1)), 1).contiguous()
            # print(y.size())(8,18,32)
            outs.append(self.decoder[i](y))
            # print(outs[i].size()) #(8,3,32)
        # B
        # print(outs.size())
        # out_mean = torch.cat(outs,0).reshape(-1,4,3,self.latent_caps_size).contiguous()
        # out_mean = torch.mean(out_mean,dim=0).contiguous()
        out_mean = torch.zeros((len(outs[i]), 3, self.latent_caps_size)).cuda()
        for i in range(len(outs)):
            out_mean = out_mean + outs[i]
        out_mean = out_mean / self.nb_primitives

        # out_mean = out_mean
        # print(out_mean.size()) ([8, 3, 32])
        # print(torch.cat(outs, 0).reshape(-1,8,3,32).size())
        # (64,4,3,32)
        # print(torch.cat(outs, 2).size()) (8,3,2048)
        # (4,3,2048)
        return torch.cat(outs, 2).contiguous(), out_mean


class PointCapsNet(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, latent_caps_size, latent_vec_size,
                 num_points):  # (1024,16,32,16,2048)
        super(PointCapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_point_caps_layer = PrimaryPointCapsLayer(prim_vec_size, num_points)
        self.capsule_groups_layer = PointGenCon(latent_vec_size + 2)
        self.latent_caps_layer = LatentCapsLayer(latent_caps_size, prim_caps_size, prim_vec_size, latent_vec_size)
        self.caps_decoder = CapsDecoder(latent_caps_size, latent_vec_size, num_points)

    def forward(self, data):
        # print("1",data.size())  (8,3,2048)
        # print("data",data)
        x1 = self.conv_layer(data)
        x2 = self.primary_point_caps_layer(x1)
        latent_capsules = self.latent_caps_layer(x2)
        # print(latent_capsules.size())  (8,32,16)
        reconstructions, cap_Group = self.caps_decoder(latent_capsules)

        # print('reconstructions size', reconstructions.size()) (B C num_points) (8,3,2048)
        # print('cap group size', cap_Group.size()) (4,3,32)
        # print("cap_Group", cap_Group)
        return latent_capsules, reconstructions, cap_Group


class PointCapsNetDecoder(nn.Module):
    def __init__(self, prim_caps_size, prim_vec_size, digit_caps_size, digit_vec_size, num_points):
        super(PointCapsNetDecoder, self).__init__()
        self.caps_decoder = CapsDecoder(digit_caps_size, digit_vec_size, num_points)

    def forward(self, latent_capsules):
        reconstructions = self.caps_decoder(latent_capsules)
        return reconstructions
