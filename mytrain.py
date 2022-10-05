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
from utils import  DUMP_IMAGES, calculate_psnr

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')

def checkpoint(net, epoch, name, save_model_path):
    os.makedirs(save_model_path, exist_ok=True)
    if epoch is not None:
        model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    else:
        model_name = 'model_{}.pth'.format(name)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    logger.info('Checkpoint saved to {} at epoch {}'.format(save_model_path, epoch))


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):
    from utils import Generator
    # prepare masks (N x C x H/2 x W/2)
    n, c, h, w = img.shape
    mask1 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    mask2 = torch.zeros(size=(n * h // 2 * w // 2 * 4, ),
                        dtype=torch.bool,
                        device=img.device)
    # prepare random mask pairs
    idx_pair = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
        dtype=torch.int64,
        device=img.device)
    rd_idx = torch.zeros(size=(n * h // 2 * w // 2, ),
                         dtype=torch.int64,
                         device=img.device)
    torch.randint(low=0,
                  high=8,
                  size=(n * h // 2 * w // 2, ),
                  generator=Generator.get_generator(),
                  out=rd_idx)
    rd_pair_idx = idx_pair[rd_idx]
    rd_pair_idx += torch.arange(start=0,
                                end=n * h // 2 * w // 2 * 4,
                                step=4,
                                dtype=torch.int64,
                                device=img.device).reshape(-1, 1)
    # get masks
    mask1[rd_pair_idx[:, 0]] = 1
    mask2[rd_pair_idx[:, 1]] = 1
    return mask1, mask2


def generate_subimages(img, mask):
    n, c, h, w = img.shape
    subimage = torch.zeros(n,
                           c,
                           h // 2,
                           w // 2,
                           dtype=img.dtype,
                           layout=img.layout,
                           device=img.device)
    # per channel
    for i in range(c):
        img_per_channel = space_to_depth(img[:, i:i + 1, :, :], block_size=2)
        img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
        subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
            n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
    return subimage


class TrainDatasetCOCOOffline(Dataset):
    def __init__(self, data_root, clean_root, corrupted_root, ann_file, patch=256, resize=True, clean_prob=0):
        super(TrainDatasetCOCOOffline, self).__init__()
        self.patch = patch

        self.data_root = data_root
        self.clean_root = clean_root
        self.corrupted_root = corrupted_root

        self.img_paths = [_['file_name'] for _ in json.load(open(ann_file))['images']]

        self.transform = transforms.ToTensor()

        self.resize_range = [256,512] if resize else None
        self.clean_prob = clean_prob

    def _get_param_crop(self, img):
        w, h = F.get_image_size(img)
        th, tw = self.patch, self.patch
        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def _get_resize_params(self, img):
        w, h = F.get_image_size(img)
        s, l = min(w,h), max(w,h)

        if self.resize_range is None:
            if s < self.patch:
                return self.patch
            return None

        lb = int(self.resize_range[0]/s*l)
        if lb <= self.resize_range[0]:
            return self.resize_range[0]
        return random.randint(self.resize_range[0], lb)

    def _get_img(self, p):
        image = Image.open(p)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        return image    

    def __getitem__(self, idx):
        img_path_clean = os.path.join(self.data_root, self.clean_root, self.img_paths[idx]) 
        img_path_corrupted = os.path.join(self.data_root, self.corrupted_root, self.img_paths[idx]) 
        imgcl = self._get_img(img_path_clean)
        imgcr = self._get_img(img_path_corrupted)
        assert imgcl.size == imgcr.size, f'{imgcl.size} {imgcr.size}'
        return self._transform_img(imgcl, imgcr)

    def _transform_img(self, imgcl, imgcr):
        # resize
        s = self._get_resize_params(imgcl)
        if s is not None:
            imgcl = F.resize(imgcl, s, F.InterpolationMode.BILINEAR, None, None)
            imgcr = F.resize(imgcr, s, F.InterpolationMode.BILINEAR, None, None)

        # random crop
        i, j, h, w = self._get_param_crop(imgcl)
        imgcl = F.crop(imgcl, i, j, h, w)
        imgcr = F.crop(imgcr, i, j, h, w)
        imgcl = self.transform(imgcl)
        imgcr = self.transform(imgcr)
        assert imgcl.shape[0] == 3, f'{img_path_clean}'
        return imgcr, imgcl

    def __len__(self):
        return len(self.img_paths)

class TrainDatasetCOCOOnline(TrainDatasetCOCOOffline):
    def __init__(self, data_root, ann_file, corruption, patch=256, fix_random_seed=True, resize=True, clean_prob=0):
        super(TrainDatasetCOCOOnline, self).__init__(data_root, None, None, ann_file, patch, resize)
        self.corruption = corruption
        self.severity = 3
        self.fix_random_seed = fix_random_seed
        self.clean_prob = clean_prob

    def __getitem__(self, idx):
        img_path_clean = os.path.join(self.data_root, self.img_paths[idx])
        imgcl = self._get_img(img_path_clean)
        # get_corruption
        if self.fix_random_seed:
            np.random.seed(idx)
        if self.corruption == 'random':
            corruption = random.choice(get_corruption_names())
        else:
            corruption = self.corruption
        arraycr = corrupt(np.array(imgcl), corruption_name=corruption, severity=self.severity)
        imgcr = Image.fromarray(arraycr)
        if self.clean_prob > 0 and random.random()<self.clean_prob:
            imgcr = imgcl
        else:
            arraycr = corrupt(np.array(imgcl), corruption_name=corruption, severity=self.severity)
            imgcr = Image.fromarray(arraycr)
        return self._transform_img(imgcl, imgcr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--clean_root', type=str)
    parser.add_argument('--corrupted_root', type=str)
    parser.add_argument('--ann_file', type=str)
    parser.add_argument('--fix_random_seed_trainset', type=int, default=1)
    parser.add_argument("--clean_prob", type=float, default=0)
    parser.add_argument('--val_ann_file', type=str)
    parser.add_argument('--val_dir', type=str)
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
    parser.add_argument('--st_val', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--patchsize', type=int, default=256)
    parser.add_argument("--Lambda1", type=float, default=1.0)
    parser.add_argument("--Lambda2", type=float, default=1.0)
    parser.add_argument("--Lambda3", type=float, default=0.0)
    parser.add_argument("--increase_ratio", type=float, default=2.0)

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
                                num_workers=2,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)
    else:
        TrainingDataset = TrainDatasetCOCOOnline(opt.data_root, opt.ann_file, opt.noisetype, patch=opt.patchsize, fix_random_seed=opt.fix_random_seed_trainset>0, resize=opt.resize_input>0, clean_prob=opt.clean_prob)
        TrainingLoader = DataLoader(dataset=TrainingDataset,
                                num_workers=2,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)

    valdataset = ValDatasetFile(opt.val_dir, opt.val_ann_file, opt.noisemethod, opt.noisetype if opt.noisetype !='random' else 'gaussian_noise')
    valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=1)

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
        cnt = 0

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        logger.info("LearningRate of Epoch {} = {}".format(epoch, current_lr))

        network.train()
        for iteration, (noisy,clean) in enumerate(TrainingLoader):
            st = time.time()

            clean = clean.cuda()
            noisy = noisy.cuda()

            optimizer.zero_grad()

            mask1, mask2 = generate_mask_pair(noisy)
            noisy_sub1 = generate_subimages(noisy, mask1)
            noisy_sub2 = generate_subimages(noisy, mask2)
            if opt.Lambda3 > 0:
                noisy_denoised = network(noisy)
            else:
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
            if opt.Lambda3 > 0:
                loss3 = torch.mean(torch.square(noisy_denoised-clean))
                loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2 + opt.Lambda3 * opt.increase_ratio * loss3
            else:
                loss3=torch.Tensor([-1])
                loss_all = opt.Lambda1 * loss1 + opt.Lambda2 * loss2

            loss_all.backward()
            optimizer.step()
            if iteration % opt.log_freq == 0:
                logger.info(
                        '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss3={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                        np.mean(loss2.item()), np.mean(loss3.item()), np.mean(loss_all.item()),
                        time.time() - st))

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
            np.random.seed(epoch*2+1)
    logger.info(f'best {best_epoch} {best_psnr}')
    
    save_model_path = os.path.join(os.path.abspath(opt.save_model_path), opt.log_name)
    if os.path.exists(f'{save_model_path}/model_best.pth'):
        os.remove(f'{save_model_path}/model_best.pth')
    os.system(f'ln -s {save_model_path}/{systime}/checkpoints/model_best.pth {save_model_path}/model_best.pth')

