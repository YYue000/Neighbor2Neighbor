from __future__ import division
import os
import time
import glob
import datetime
import argparse
import numpy as np

import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F

from arch_unet import UNet
from validate import validate

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./Imagenet_val')
    parser.add_argument('--val_dir', type=str, default='./validation')
    parser.add_argument('--save_model_path', type=str, default='./results')
    parser.add_argument('--log_name', type=str, default='unet_gauss25_b4e100r02')
    parser.add_argument('--gpu_devices', default='0', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--n_snapshot', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--patchsize', type=int, default=256)
    parser.add_argument("--Lambda1", type=float, default=1.0)
    parser.add_argument("--Lambda2", type=float, default=1.0)
    parser.add_argument("--increase_ratio", type=float, default=2.0)

    opt, _ = parser.parse_known_args()
    systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    operation_seed_counter = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices


    # Training Set
    TrainingDataset = DataLoader_Imagenet_val(opt.data_dir, patch=opt.patchsize)
    TrainingLoader = DataLoader(dataset=TrainingDataset,
                                num_workers=8,
                                batch_size=opt.batchsize,
                                shuffle=True,
                                pin_memory=False,
                                drop_last=True)

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

    checkpoint(network, 0, "model")
    logger.info('init finish')

    validate_opt = EasyDict({''})

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

            loss_all.backward()
            optimizer.step()
            logger.info(
                '{:04d} {:05d} Loss1={:.6f}, Lambda={}, Loss2={:.6f}, Loss_Full={:.6f}, Time={:.4f}'
                .format(epoch, iteration, np.mean(loss1.item()), Lambda,
                        np.mean(loss2.item()), np.mean(loss_all.item()),
                        time.time() - st))

        scheduler.step()

        if epoch % opt.n_snapshot == 0 or epoch == opt.n_epoch:
            # save checkpoint
            checkpoint(network, epoch, "model")
            # validation
            save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
            np.random.seed(101)
            avg_psnr, avg_ssim = validate(network, validate_opt)
            logger.info("epoch:{},avg_psnr{},avg_ssim{}\n".format(epoch, avg_psnr, avg_ssim))

def checkpoint(net, epoch, name):
    save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
    os.makedirs(save_model_path, exist_ok=True)
    model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
    save_model_path = os.path.join(save_model_path, model_name)
    torch.save(net.state_dict(), save_model_path)
    print('Checkpoint saved to {}'.format(save_model_path))


def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)


def generate_mask_pair(img):
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
                  generator=get_generator(),
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


class DataLoader_Imagenet_val(Dataset):
    def __init__(self, data_root, clean_root, corrupted_root, ann_file, patch=256):
        super(DataLoader_Imagenet_val, self).__init__()
        self.patch = patch

        self.data_root = data_root
        self.clean_root = clean_root
        self.corrupted_root = corrupted_root

        self.img_paths = [_['file_name'] for _ in json.load(open(ann_file))['images']]

        self.transform = transforms.ToTensor()]

    def _get_param_crop(self, img):
        w, h = img.size
        th, tw = self.patch, self.patch
        if w >= tw and h >= th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __getitem__(self, index):
        _get_img = lambda p: Image.open(p) #.convert("RGB")

        img_path_clean = os.path.join(self.data_root, self.clean_root, self.img_paths[idx]) 
        img_path_corrupted = os.path.join(self.data_root, self.clean_root, self.img_paths[idx]) 
        imgcl = _get_img(img_path_clean)
        imgcr = _get_img(img_path_corrupted)
        assert imgcl.size == imgcr.size, f'{imgcl.size} {imgcr.size}'

        # random crop
        i, j, h, w = self._get_param_crop(imgcl)
        imgcl = F.crop(imgcl, i, j, h, w)
        imgcr = F.crop(imgcr, i, j, h, w)
        imgcl = self.transformer(imgcl)
        imgcr = self.transformer(imgcr)
        return imgcr, imgcl

    def __len__(self):
        return len(self.img_paths)


