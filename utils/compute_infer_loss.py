import torch
from model.loss import *
from model.model_util import mean
from PerceptualSimilarity.models import compare_ssim as SSIM
import numpy as np


class compute_infer_loss:
    def __init__(self, infer_loss_fname, LPIPS_net_type='alex'):
        self.infer_loss_fname = infer_loss_fname
        self.perceptual_loss_fn = perceptual_loss(net=LPIPS_net_type)
        self.mse_loss_fn = l2_loss()
        self.ssim_loss_fn = SSIM
        self.loss = {'perceptual_loss': [],
                     'mse_loss': [],
                     'ssim_loss': []}

    def __call__(self, pred_img, gt_img):
        self.loss['perceptual_loss'].append(self.perceptual_loss_fn(pred_img, gt_img).item())
        self.loss['mse_loss'].append(self.mse_loss_fn(pred_img, gt_img).item())
        self.loss['ssim_loss'].append(self.ssim_loss_fn(pred_img.squeeze().cpu().numpy(), 
                                                        gt_img.squeeze().cpu().numpy()))

    def write_loss(self):
        mean_lpips = mean(self.loss['perceptual_loss'])
        mean_mse = mean(self.loss['mse_loss'])
        mean_ssim = mean(self.loss['ssim_loss'])

        with open(self.infer_loss_fname, 'w') as f:
            f.write('perceptual loss for each step:{}\n'.format(self.loss['perceptual_loss']))
            f.write('mse loss for each step:{}\n'.format(self.loss['mse_loss']))
            f.write('ssim loss for each step:{}\n'.format(self.loss['ssim_loss']))
            f.write('******************************\n')
            f.write('mean perceptual loss for whole sequences:{}\n'.format(mean_lpips))
            f.write('mean mse loss for whole sequences:{}\n'.format(mean_mse))
            f.write('mean ssim loss for whole sequences:{}\n'.format(mean_ssim))
            
        return {'mean_lpips': mean_lpips, 'mean_mse': mean_mse, 'mean_ssim': mean_ssim}
