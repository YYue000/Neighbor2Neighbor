import numpy as np
import cv2
from PIL import Image
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from imagecorruptions import corrupt

from arch_unet import UNet
from utils import AugmentNoise, calculate_ssim, calculate_psnr

import argparse
import random
import os


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')


def read_image(p):
    return np.asarray(Image.open(p)).astype(np.float32)

class ValDataset(Dataset):
    def __init__(self, data_root, noise_method=None, noise_type=None):
        super(ValDataset, self).__init__()
        img_paths = []
        for r, d, files in os.walk(data_root):
            for f in files:
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPEG'):
                    img_paths.append(os.path.join(r, f))
        self.img_paths = img_paths
        
        logger.info(f'noisetype {noise_method} {noise_type}')
        self.noise_method = noise_method
        if noise_method == 'AugmentNoise':
            self.noise_adder = AugmentNoise(style=noise_type)
        elif noise_method == 'imagecorruptions':
            self.severity = 3
            self.corruption = self.noisy_type
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _add_noise(self, im):
        if self.noise_method is None:
            return im
        elif self.noise_method == 'AugmentNoise':
            im = im/255.0
            noisy_im = self.noise_adder.add_valid_noise(im)
            noisy255 = np.clip(noisy_im * 255.0 + 0.5, 0, 255).astype(np.uint8)
            return noisy255
        elif self.noise_method == 'imagecorruptions':
            return corrupt(im.astype(np.uint8), corruption_name=self.noise_type, severity=self.severity)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        imgpath = self.img_paths[idx]

        im = read_image(imgpath)

        noisy_im = self._add_noise(im)

        # padding to square
        H = noisy_im.shape[0]
        W = noisy_im.shape[1]
        assert H== im.shape[0] and W == im.shape[1]
        val_size = (max(H, W) + 31) // 32 * 32
        noisy_im = np.pad(
            noisy_im,
            [[0, val_size - H], [0, val_size - W], [0, 0]],
            'reflect').astype(np.uint8)
        noisy_im = self.transform(noisy_im)
        im = self.transform(im.astype(np.uint8))
        return imgpath.split('/')[-1], noisy_im, im 

    def __len__(self):
        return len(self.img_paths)


def validate(network, valdataloader, opt, verbose=True):
    network.eval()

    validation_path = os.path.join(opt.save_model_path, "validation")
    os.makedirs(validation_path, exist_ok=True)
    np.random.seed(101)
    

    psnr_result = []
    ssim_result = []
    for im_name, noisy_im, im in valdataloader:
        im_name = im_name[0]
        if verbose:
            logger.info(f'process {im_name}')

        noisy_im = noisy_im.cuda()
        H, W = im.shape[2:]

        origin255 = (im.permute(0,2,3,1).squeeze().numpy()*255).astype(np.uint8)

        with torch.no_grad():
            prediction = network(noisy_im)
            prediction = prediction[:, :, :H, :W]
        prediction = prediction.permute(0,2,3,1)
        prediction = prediction.cpu().data.clamp(0, 1).numpy()
        prediction = prediction.squeeze()
        pred255 = np.clip(prediction * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # calculate psnr
        cur_psnr = calculate_psnr(origin255.astype(np.float32),
                                  pred255.astype(np.float32))
        psnr_result.append(cur_psnr)
        cur_ssim = calculate_ssim(origin255.astype(np.float32),
                                  pred255.astype(np.float32))
        ssim_result.append(cur_ssim)

        save_path = os.path.join(
            validation_path,
            "{}_clean.jpg".format(im_name.split('.')[0]))
        if opt.noisetype is not None:
            Image.fromarray(origin255).convert('RGB').save(save_path)

        save_path = save_path.replace('clean', 'noisy')
        if not opt.dump_denoise_only:
            noisy255 = noisy_im[:,:,:H,:W]
            noisy255 = (noisy255.permute(0,2,3,1).squeeze().cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(noisy255).convert('RGB').save(save_path)
        if not opt.dump_denoise_only:
            save_path = save_path.replace('noisy', 'denoised')
            Image.fromarray(pred255).convert('RGB').save(save_path)
        else:
            save_path = save_path.replace('_noisy', '')
            Image.fromarray(pred255).convert('RGB').save(save_path)

    avg_psnr = np.mean(psnr_result)
    avg_ssim = np.mean(ssim_result)
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--noisemethod", type=str, default=None)
    parser.add_argument("--noisetype", type=str, default=None)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--save_model_path', type=str, default='./results')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=3)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--dump_denoise_only", action="store_true")

    opt = parser.parse_args()


    network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
    network = network.cuda()
    network.load_state_dict(torch.load(opt.ckpt))

    valdataset = ValDataset(opt.val_dir, opt.noisemethod, opt.noisetype)
    valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False)

    _,__ = validate(network, valdataloader, opt)
    logger.info(f'{_} {__}')


