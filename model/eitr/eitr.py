import torch
import torch.nn.functional as F
from torch import nn

from model.model_util import CropSize
from .u_trans import mls_tpa


class EITR(mls_tpa):
    def __init__(self, eitr_kwargs):
        super().__init__(eitr_kwargs['num_bins'], eitr_kwargs['norm'])

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
                 N x 1 x H x W
        """
        n, c, H, W = event_tensor.size()

        # pad size
        factor = {'h':8, 'w':8}
        pad_crop = CropSize(W, H, factor)
        if (H % factor['h'] != 0) or (W % factor['w'] != 0):
            event_tensor = pad_crop.pad(event_tensor)

        out = self.func(event_tensor)

        # crop size
        if (H % factor['h'] != 0) or (W % factor['w'] != 0):
            out = pad_crop.crop(out)

        return {'image': out}

