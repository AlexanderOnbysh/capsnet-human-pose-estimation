import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ITOPDataset
from models.chamfer_distance import ChamferDistance
from utils import get_data_loaders, load_network, get_lr, build_map_plot, thresholds

writer = SummaryWriter('runs/normalization-long-run-2000-points')
chamfer_dist = ChamferDistance()


class TrainingPipeline:
    pass

    def __init__(self, run_name: str):
        self._name = run_name
        self._prepare_directory()

    def _prepare_directory(self):
        if not os.path.exists(opt.outf):
            os.makedirs(opt.outf)
        pass


def main():
    train_dataloader, test_dataloader = get_data_loaders(opt)
    capsule_net = load_network(opt)

    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.Adam(capsule_net.parameters(), lr=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    capsule_net.train()

    if opt.load_weights != '':
        print('Load weights from', opt.load_weights)
        capsule_net.module.load_state_dict(torch.load(opt.load_weights))

    # =============== TRAIN =========================
    global_train_step = 0
    global_test_step = 0
    best_map = 0
    for i, epoch in enumerate(range(opt.nepoch)):
        train_loss_sum = 0
        true_values_per_joint_per_threshold = np.zeros((15, thresholds.shape[0]))
        total_examples = 0
        capsule_net.train()
        print(f'======>>>>> Online epoch: {epoch}, lr={get_lr(optimizer)} <<<<<======')
        for data in tqdm(train_dataloader):
            global_train_step += 1
            points, coords, mean, maxv = data
            points, coords = points.cuda(non_blocking=True), coords.cuda(non_blocking=True)
            if points.size(0) < opt.batchSize:
                break

            optimizer.zero_grad()
            estimation, reconstructions = capsule_net(points)

            dist1, dist2 = chamfer_dist(reconstructions, points)
            reconstruction_loss = (torch.mean(dist1)) + (torch.mean(dist2))
            regression_loss = criterion(estimation, coords)
            total_loss = reconstruction_loss + regression_loss
            total_loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            writer.add_scalar('train/reconstruction-loss', reconstruction_loss.detach().cpu().numpy(),
                              global_step=global_train_step)
            writer.add_scalar('train/regression-loss', regression_loss.detach().cpu().numpy(),
                              global_step=global_train_step)

            points_to_display = np.expand_dims(points.cpu().numpy()[0], axis=0)
            reconstructions_to_display = np.expand_dims(reconstructions.detach().cpu().numpy()[0], axis=0)
            points_to_display = ITOPDataset.denormalize(points_to_display, mean[0].numpy(), maxv[0].numpy())
            reconstructions_to_display = ITOPDataset.denormalize(reconstructions_to_display, mean[0].numpy(),
                                                                 maxv[0].numpy())

            writer.add_mesh('train/points', vertices=points_to_display, global_step=global_train_step)
            writer.add_mesh('train/points-reconstruction', vertices=reconstructions_to_display,
                            global_step=global_train_step)
            writer.add_scalar('train/lr', get_lr(optimizer), global_step=global_train_step)

            # mAP calculating
            total_examples += len(estimation)
            estimation = ITOPDataset.denormalize(estimation.detach().cpu().numpy(), mean.numpy(), maxv.numpy())
            coords = ITOPDataset.denormalize(coords.detach().cpu().numpy(), mean.numpy(), maxv.numpy())
            batch_diff = np.linalg.norm(estimation - coords, axis=2)  # N x JOINT_SIZE
            for example in batch_diff:
                for i, joint_diff in enumerate(example):
                    true_values_per_joint_per_threshold[i] += (joint_diff < thresholds).astype(int)

            train_loss_sum += total_loss.item()

        scheduler.step()
        torch.cuda.synchronize()

        map_fig, map_01 = build_map_plot(thresholds, true_values_per_joint_per_threshold / total_examples)
        writer.add_figure('train/map', map_fig, global_step=global_train_step)

        # =============== EVAL =========================

        test_reconstruction_loss_sum = 0
        test_regression_loss_sum = 0
        true_values_per_joint_per_threshold = np.zeros((15, thresholds.shape[0]))
        total_examples = 0
        for i, data in enumerate(tqdm(test_dataloader)):
            global_test_step += 1
            capsule_net.eval()
            points, coords, mean, maxv = data
            points, coords = points.cuda(), coords.cuda()

            estimation, reconstructions = capsule_net(points)
            dist1, dist2 = chamfer_dist(points, reconstructions)
            test_reconstruction_loss = (torch.mean(dist1)) + (torch.mean(dist2))
            test_regression_loss = criterion(estimation, coords)
            test_reconstruction_loss_sum += test_reconstruction_loss.item()
            test_regression_loss_sum += test_regression_loss.item()

            points_to_display = np.expand_dims(points.cpu().numpy()[0], axis=0)
            reconstructions_to_display = np.expand_dims(reconstructions.detach().cpu().numpy()[0], axis=0)
            points_to_display = ITOPDataset.denormalize(points_to_display, mean[0].numpy(), maxv[0].numpy())
            reconstructions_to_display = ITOPDataset.denormalize(reconstructions_to_display, mean[0].numpy(),
                                                                 maxv[0].numpy())
            writer.add_mesh('test/points', vertices=points_to_display, global_step=global_test_step)
            writer.add_mesh('test/points-reconstruction', vertices=reconstructions_to_display,
                            global_step=global_test_step)

            # -------- mAP calculating
            total_examples += len(estimation)

            estimation = ITOPDataset.denormalize(estimation.detach().cpu().numpy(), mean.numpy(), maxv.numpy())
            coords = ITOPDataset.denormalize(coords.detach().cpu().numpy(), mean.numpy(), maxv.numpy())
            batch_diff = np.linalg.norm(estimation - coords, axis=2)  # N x JOINT_SIZE
            for example in batch_diff:
                for i, joint_diff in enumerate(example):
                    true_values_per_joint_per_threshold[i] += (joint_diff < thresholds).astype(int)

        avg_reconstruction = test_reconstruction_loss_sum / len(test_dataloader)
        avg_regression = test_regression_loss_sum / len(test_dataloader)
        writer.add_scalar('test/reconstruction-loss', avg_reconstruction, global_step=global_test_step)
        writer.add_scalar('test/regression-loss', avg_regression, global_step=global_test_step)

        map_fig, map_01 = build_map_plot(thresholds, true_values_per_joint_per_threshold / total_examples)
        writer.add_figure('test/map', map_fig, global_step=global_test_step)

        if best_map < map_01:
            best_map = map_01
            torch.save(capsule_net.module.state_dict(), f'{save_dir}/{epoch:03}-{best_map:0.3}-capsule_net-module.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2000, help='input point set size')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')

    parser.add_argument('--save_root_dir', type=str, default='results', help='output folder')

    parser.add_argument('--batchSize', type=int, default=24, help='input batch size')  #
    parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')

    parser.add_argument('--JOINT_NUM', type=int, default=15, help='number of joints')
    parser.add_argument('--INPUT_FEATURE_NUM', type=int, default=3, help='number of input point features')
    parser.add_argument('--PCA_SZ', type=int, default=45, help='number of PCA components')

    parser.add_argument('--network_train', type=str, default='all',
                        help='What part of network to train (all, capsnet, regression)')
    parser.add_argument('--load_weights', type=str, default='', help='Path to weights')
    opt = parser.parse_args()
    save_dir = os.path.join(opt.save_root_dir)  # results/P0
    os.makedirs(save_dir, exist_ok=True)
    print(opt)

    main()
