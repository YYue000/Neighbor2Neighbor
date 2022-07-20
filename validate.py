import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from imagecorruptions import corrupt

from arch_unet import UNet
from utils import AugmentNoise, calculate_ssim, calculate_psnr, DUMP_IMAGES

import argparse
import random
import os
import json
from easydict import EasyDict


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s#%(lineno)d:%(message)s')
logger = logging.getLogger('global')

def read_image(p):
    image = Image.open(p)
    if image.mode != 'RGB':
        image = image.convert("RGB")

    return np.array(image).astype(np.float32)
    #return np.asarray(image).astype(np.float32)

class ValDatasetDir(Dataset):
    def __init__(self, data_root, noise_method=None, noise_type=None):
        super(ValDatasetDir, self).__init__()
        self.data_root = data_root
        self.img_paths = self._get_img_paths()
        
        logger.info(f'noisetype {noise_method} {noise_type}')
        self.noise_method = noise_method
        if noise_method == 'AugmentNoise':
            self.noise_adder = AugmentNoise(style=noise_type)
        elif noise_method == 'imagecorruptions':
            self.severity = 3
            self.corruption = noise_type
        self.transform = transforms.Compose([transforms.ToTensor()])

    def _get_img_paths(self):
        img_paths = []
        for r, d, files in os.walk(self.data_root):
            for f in files:
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.JPEG'):
                    img_paths.append(os.path.join(r, f))
        return img_paths

    def _add_noise(self, im):
        if self.noise_method is None:
            return im.astype(np.uint8)
        elif self.noise_method == 'AugmentNoise':
            im = im/255.0
            noisy_im = self.noise_adder.add_valid_noise(im)
            noisy255 = np.clip(noisy_im * 255.0 + 0.5, 0, 255).astype(np.uint8)
            return noisy255
        elif self.noise_method == 'imagecorruptions':
            return corrupt(im.astype(np.uint8), corruption_name=self.corruption, severity=self.severity)
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

class ValDatasetFile(ValDatasetDir):
    def __init__(self, data_root, ann_file, noise_method=None, noise_type=None):
        self.ann_file = ann_file
        self.data_root = data_root
        super(ValDatasetFile, self).__init__(data_root, noise_method, noise_type)

    def _get_img_paths(self):
         return [os.path.join(self.data_root, _['file_name']) for _ in json.load(open(self.ann_file))['images']]


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
        if opt.dump_images in [DUMP_IMAGES.DENOISED_CLEAN, DUMP_IMAGES.DENOISED_NOISY_CLEAN]:
            Image.fromarray(origin255).convert('RGB').save(save_path)

        save_path = save_path.replace('_clean.jpg', '_noisy.jpg')
        if opt.dump_images in [DUMP_IMAGES.DENOISED_NOISY, DUMP_IMAGES.DENOISED_NOISY_CLEAN]:
            noisy255 = noisy_im[:,:,:H,:W]
            noisy255 = (noisy255.permute(0,2,3,1).squeeze().cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(noisy255).convert('RGB').save(save_path)
            save_path = save_path.replace('_noisy.jpg', '_denoised.jpg')
        else:
            save_path = save_path.replace('_noisy.jpg', '.jpg')
       
        if opt.dump_images != DUMP_IMAGES.NO_DUMP:
            Image.fromarray(pred255).convert('RGB').save(save_path)

    avg_psnr = np.mean(psnr_result)
    avg_ssim = np.mean(ssim_result)
    if verbose:
        logger.info(f'psnr {psnr_result}')
        logger.info(f'ssim {ssim_result}')
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--noisemethod", type=str, default=None)
    parser.add_argument("--noisetype", type=str, default=None)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--val_ann_file', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default='./results')
    parser.add_argument('--n_feature', type=int, default=48)
    parser.add_argument('--n_channel', type=int, default=3)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--dump_images", type=str, choices=list(DUMP_IMAGES))

    opt = parser.parse_args()
    opt.dump_images = DUMP_IMAGES[opt.dump_images]
    logger.info(f'{opt}')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


    network = UNet(in_nc=opt.n_channel,
               out_nc=opt.n_channel,
               n_feature=opt.n_feature)
    network = network.cuda()
    network.load_state_dict(torch.load(opt.ckpt))

    if opt.val_ann_file is None:
        valdataset = ValDatasetDir(opt.val_dir, opt.noisemethod, opt.noisetype)
    else:
        valdataset = ValDatasetFile(opt.val_dir, opt.val_ann_file, opt.noisemethod, opt.noisetype)
    valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False)

    _,__ = validate(network, valdataloader, opt, verbose=opt.verbose)
    logger.info(f'avg{_} {__}')


