import torch
import torch.nn as nn

from models.capsulenet.pointcapsnet import PointCapsNet

nstates_plus_1 = [64, 512, 1024]
nstates_plus_2 = [128, 256]
nstates_plus_3 = [256, 512, 1024, 512]
nstates_plus_4 = [1024, 512, 256]


class Capsule_handnet(nn.Module):
    def __init__(self, opt):
        super(Capsule_handnet, self).__init__()
        self.batchSize = opt.batchSize
        self.latent_caps_size = opt.latent_caps_size
        self.num_outputs = opt.PCA_SZ
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM

        if opt.network_train == 'capsnet':
            self._initialize_regression(opt)
            for p in self.parameters():
                p.requires_grad = False
            self._initialize_capsnet(opt)

        if opt.network_train == 'regression':
            self._initialize_capsnet(opt)
            for p in self.parameters():
                p.requires_grad = False
            self._initialize_regression(opt)
        else:
            self._initialize_capsnet(opt)
            self._initialize_regression(opt)

    def _initialize_capsnet(self, opt):
        self.Capsulenet1 = PointCapsNet(opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size,
                                        opt.latent_vec_size, opt.num_points)

    def _initialize_regression(self, opt):
        self.netR_0 = nn.Sequential(
            nn.MaxPool1d(16, 1)
        )

        self.netR_1 = nn.Sequential(
            torch.nn.Conv1d(nstates_plus_1[0], nstates_plus_1[1], 1),
            torch.nn.BatchNorm1d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            torch.nn.Conv1d(nstates_plus_1[1], nstates_plus_1[2], 1),
            torch.nn.BatchNorm1d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(64, 1)
        )

        self.netR_2 = nn.Sequential(
            torch.nn.Conv1d(nstates_plus_1[2] + 64, nstates_plus_1[2], 1),
            torch.nn.BatchNorm1d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(64, 1)
        )

        self.netR_3 = nn.Sequential(
            torch.nn.Conv1d(nstates_plus_3[0], nstates_plus_3[1], 1),
            torch.nn.BatchNorm1d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            torch.nn.Conv1d(nstates_plus_3[1], nstates_plus_3[2], 1),
            torch.nn.BatchNorm1d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(64, 1)
        )

        self.netR_FC = nn.Sequential(
            nn.Linear(nstates_plus_4[0], nstates_plus_4[1]),
            nn.BatchNorm1d(nstates_plus_4[1]),
            nn.ReLU(inplace=True),

            nn.Linear(nstates_plus_4[1], nstates_plus_4[2]),
            nn.BatchNorm1d(nstates_plus_4[2]),
            nn.ReLU(inplace=True)
        )
        self.linear_fc = nn.Linear(nstates_plus_4[2], self.num_outputs)

    def forward(self, x):
        x = x.transpose(1, 2)

        x, reconstructions, _ = self.Capsulenet1(x)

        # TODO: why do we need this
        # x = x.transpose(2, 1).contiguous()  # (B*64*64)

        temp = x.detach().cuda()
        x = self.netR_1(x)
        x = x.expand(x.shape[0], nstates_plus_1[2], self.latent_caps_size)
        x = torch.cat((temp, x), 1).contiguous()
        x = self.netR_2(x)
        x = x.view(-1, nstates_plus_3[2])
        x = self.netR_FC(x)
        x = self.linear_fc(x)
        return x.reshape(-1, 15, 3), reconstructions.transpose(2, 1)
