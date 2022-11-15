from __future__ import division
import os
import time
import glob
import datetime
import argparse
import json
from easydict import EasyDict
import random

import cv2
from PIL import Image
import numpy as np
import math
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from imagecorruptions import corrupt, get_corruption_names

from arch_unet import UNet
from validate import validate, ValDatasetFile
from utils import Generator, DUMP_IMAGES, calculate_psnr
from mytrain import TrainDatasetCOCOOffline, TrainDatasetCOCOOnline, checkpoint

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--clean_root', type=str)
    parser.add_argument('--corrupted_root', type=str)
    parser.add_argument('--ann_file', type=str)
    parser.add_argument('--fix_random_seed_trainset', type=int, default=1)
    parser.add_argument('--val_ann_file', type=str)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--val_clean_dir', type=str, default=None)
    parser.add_argument("--clean_prob", type=float, default=0)
    parser.add_argument("--noisemethod", type=str, default=None)
    parser.add_argument("--noisetype", type=str, default=None)
    parser.add_argument("--dump_images", type=str, default=DUMP_IMAGES.DENOISED_NOISY_CLEAN, choices=list(DUMP_IMAGES))
    parser.add_argument("--resize_input", type=int, default=1)
    parser.add_argument('--save_model_path', type=str, default='./results')
    parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
    parser.add_argument('--log_freq', type=int, default=1)
    parser.add_argument('--gpu_devices', default='0', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--n_snapshot', type=int, default=1)
    parser.add_argument('--n_val', type=int, default=None)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--patchsize', type=int, default=256)
    parser.add_argument("--Lambda", type=float, default=1.0)
    parser.add_argument('--st_val', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--val_num_workers', type=int, default=2)


    opt, _ = parser.parse_known_args()
    opt.dump_images = DUMP_IMAGES[opt.dump_images]
    if opt.n_val is None:
        opt.n_val = opt.n_snapshot
    systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    operation_seed_counter = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices

    
    logging.basicConfig(filename = os.path.join(opt.save_model_path, opt.log_name, systime, "train.log"),
                    filemode = "w",
                    format = '%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
    logger.info(f'{opt}')

    # Training Set
    if opt.clean_root is not None:
        TrainingDataset = TrainDatasetCOCOOffline(opt.data_root, opt.clean_root, opt.corrupted_root, opt.ann_file, patch=opt.patchsize, resize=opt.resize_input>0, clean_prob=opt.clean_prob)
        TrainingLoader = DataLoader(dataset=TrainingDataset,
                                num_workers=opt.num_workers,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)
    else:
        TrainingDataset = TrainDatasetCOCOOnline(opt.data_root, opt.ann_file, opt.noisetype, patch=opt.patchsize, fix_random_seed=opt.fix_random_seed_trainset>0, resize=opt.resize_input>0, clean_prob=opt.clean_prob)
        TrainingLoader = DataLoader(dataset=TrainingDataset,
                                num_workers=opt.num_workers,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)

    valdataset = ValDatasetFile(opt.val_dir, opt.val_clean_dir, opt.val_ann_file, opt.noisemethod, opt.noisetype if opt.noisetype !='random' else 'gaussian_noise')
    valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=opt.val_num_workers)

    # Network
    network = UNet(in_nc=opt.n_channel,
                   out_nc=opt.n_channel,
                   n_feature=opt.n_feature)
    if opt.parallel:
        network = torch.nn.DataParallel(network)
    network = network.cuda()

    # about training scheme
    num_epoch = opt.n_epoch
    ratio = num_epoch / 100
    optimizer = optim.Adam(network.parameters(), lr=opt.lr)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[
                                             int(20 * ratio) - 1,
                                             int(40 * ratio) - 1,
                                             int(60 * ratio) - 1,
                                             int(80 * ratio) - 1
                                         ],
                                         gamma=opt.gamma)
    logger.info("Batchsize={}, number of epoch={}".format(opt.batchsize, opt.n_epoch))

    #checkpoint(network, 0, "model")
    logger.info('init finish')

    validate_opt = EasyDict({'noisemethod': opt.noisemethod, 
        'noisetype': opt.noisetype,
        'dump_images': opt.dump_images})

    best_psnr = -1
    best_epoch = 0

    for epoch in range(1, opt.n_epoch + 1):
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        logger.info("LearningRate of Epoch {} = {}".format(epoch, current_lr))

        network.train()
        for iteration, (noisy,clean) in enumerate(TrainingLoader):
            st = time.time()

            clean = clean.cuda()
            noisy = noisy.cuda()

            optimizer.zero_grad()
            

            """
            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)
            with torch.no_grad():
                noisy_denoised = network(noisy)
            noisy_sub1_denoised = generate_subimages(noisy_denoised, mask1)
            noisy_sub2_denoised = generate_subimages(noisy_denoised, mask2)

            noisy_output = network(noisy_sub1)
            noisy_target = noisy_sub2
            Lambda = epoch / opt.n_epoch * opt.increase_ratio
            diff = noisy_output - noisy_target
            exp_diff = noisy_sub1_denoised - noisy_sub2_denoised

            loss1 = torch.mean(diff**2)
            loss2 = Lambda * torch.mean((diff - exp_diff)**2)
            loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2
            """

            output = network(noisy)
            loss_all = opt.Lambda*torch.mean(torch.square(output-clean))

            loss_all.backward()
            optimizer.step()
            if iteration % opt.log_freq == 0:
                psnr = calculate_psnr(output.detach().cpu().numpy(), clean.cpu().numpy())
                logger.info(
                        '{:04d} {:05d} Loss_Full={:.6f}, Time={:.4f} psnr={:.2f}'
                .format(epoch, iteration, loss_all.item(),
                        time.time() - st, psnr))

        scheduler.step()

        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
            # save checkpoint
            save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime, 'checkpoints')
            checkpoint(network, epoch, "model", save_model_path)
        if epoch >= opt.st_val and epoch % opt.n_val == 0 or epoch == opt.n_epoch:
            # validation
            save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
            np.random.seed(101)
            validate_opt['save_model_path'] = save_model_path
            avg_psnr, avg_ssim = validate(network, valdataloader, validate_opt, verbose=False)
            logger.info("epoch:{},avg_psnr{},avg_ssim{}\n".format(epoch, avg_psnr, avg_ssim))
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime, 'checkpoints')
                checkpoint(network, None, "best", save_model_path)
    logger.info(f'best {best_epoch} {best_psnr}')

    save_model_path = os.path.join(os.path.abspath(opt.save_model_path), opt.log_name)
    if os.path.exists(f'{save_model_path}/model_best.pth'):
        os.remove(f'{save_model_path}/model_best.pth')
    os.system(f'ln -s {save_model_path}/{systime}/checkpoints/model_best.pth {save_model_path}/model_best.pth')
