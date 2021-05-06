import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from matplotlib import pyplot as plt

from consts import DATASET_PATH
from dataset import ITOPDataset
from models.network import Capsule_handnet


def get_data_loaders(opt):
    train_dataloader = torch.utils.data.DataLoader(ITOPDataset(root_path=DATASET_PATH,
                                                               train=True),
                                                   batch_size=opt.batchSize,
                                                   shuffle=True,
                                                   num_workers=int(opt.workers),
                                                   pin_memory=True,
                                                   drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(ITOPDataset(root_path=DATASET_PATH,
                                                              train=False),
                                                  batch_size=opt.batchSize,
                                                  shuffle=False,
                                                  num_workers=int(opt.workers),
                                                  pin_memory=True,
                                                  drop_last=True)

    return train_dataloader, test_dataloader


def load_network(opt):
    capsule_net = Capsule_handnet(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capsule_net = capsule_net.cuda()
    capsule_net.to(device)
    capsule_net = torch.nn.DataParallel(capsule_net)

    return capsule_net


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def build_map_plot(x, y):
    y = y.mean(axis=0)

    index_threshold_01 = np.where(thresholds == 0.1)[0][0]
    map_at_01 = y[index_threshold_01]

    fig = plt.figure()

    plt.xlabel('allowed error distance (in meters)')
    plt.ylabel('mAP')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.plot(x, y)
    plt.scatter([0.1], [map_at_01])
    plt.annotate(f'{map_at_01:.02}', (0.1, map_at_01))

    return fig, map_at_01


thresholds = np.arange(0, 1.01, 0.01)
