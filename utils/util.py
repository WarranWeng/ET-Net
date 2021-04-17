import json
import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from math import fabs, ceil, floor
from torch.nn import ZeroPad2d
from os.path import join
import torch


def minmax_normalization(image, device):
    mini = np.percentile(torch.flatten(image).cpu().detach().numpy(), 1)
    maxi= np.percentile(torch.flatten(image).cpu().detach().numpy(), 99)
    image_morm = (image - mini) / (maxi - mini + 1e-5)
    image_morm = torch.clamp(image_morm, 0, 1)
    
    return image_morm.to(device)


class attn_weight_hook():
    """
    A class for plotting attention weight mapps of encoder and decoder from the transformer 

    """
    def __init__(self, model: torch.nn.Module, is_trans: bool = False):
        self.model = model
        self.is_trans = is_trans

    def reset_list(self):
        if self.is_trans:
            self.conv_features, self.dec_attn_weights, self.enc_attn_weights = [], [], []

    def set_hook(self):
        if self.is_trans:
            self.hooks = [
                self.model.encoder_backbone.register_forward_hook(
                    lambda inmediate, input, output: self.conv_features.append(output.cpu())
                ),
                self.model.dual_subdecoder_transformer.spa_dec.layers[-1].multihead_attn.register_forward_hook(
                    lambda inmediate, input, output: self.dec_attn_weights.append(output[1].cpu())
                ),
                self.model.dual_subdecoder_transformer.spa_enc.layers[-1].self_attn.register_forward_hook(
                    lambda inmediate, input, output: self.enc_attn_weights.append(output[1].cpu())
                )
            ]

    def remove_hook(self):
        if self.is_trans:
            for hook in self.hooks:
                hook.remove() 

    def plot_attn_map(self, index, img, save_dir: str):
        if self.is_trans:
            path_to_enc_satten = join(save_dir, 'enc_satten')
            path_to_dec_catten = join(save_dir, 'dec_catten')

            if not os.path.exists(path_to_enc_satten) or not os.path.exists(path_to_enc_satten):
                os.mkdir(path_to_enc_satten)
                os.mkdir(path_to_dec_catten)

            self.conv_features = self.conv_features[0]
            self.enc_attn_weights = self.enc_attn_weights[0]
            self.dec_attn_weights = self.dec_attn_weights[0]

            img = img.squeeze().unsqueeze(-1)

            idxs = [(25, 80), (50, 75), (90, 120), (150, 32)] # points_of_interest
            fact = 8
            shape = self.conv_features.shape[-2:]
            enc_sattn = self.enc_attn_weights[0].reshape(shape + shape)
            dec_sattn = self.dec_attn_weights[0].reshape(shape + shape)

            # plot encoder self attention map
            fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
            gs = fig.add_gridspec(2, 4)
            axs = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[0, -1]),
                fig.add_subplot(gs[1, -1]),
            ]
            for idx_o, ax in zip(idxs, axs):
                idx = (idx_o[0] // fact, idx_o[1] // fact)
                ax.imshow(enc_sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'encoder self-attention{idx_o}')

            fcenter_ax = fig.add_subplot(gs[:, 1:-1])
            fcenter_ax.imshow(img, cmap ='gray')
            for (y, x) in idxs:
                # x = ((x // fact) + 0.5) * fact
                # y = ((y // fact) + 0.5) * fact
                fcenter_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
                fcenter_ax.axis('off')

            fname = f'frame{index}_enc_satten.png'
            plt.savefig(join(path_to_enc_satten, fname))

            # plot decoder cross attention map
            fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
            gs = fig.add_gridspec(2, 4)
            axs = [
                fig.add_subplot(gs[0, 0]),
                fig.add_subplot(gs[1, 0]),
                fig.add_subplot(gs[0, -1]),
                fig.add_subplot(gs[1, -1]),
            ]
            for idx_o, ax in zip(idxs, axs):
                idx = (idx_o[0] // fact, idx_o[1] // fact)
                ax.imshow(dec_sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'decoder cross-attention{idx_o}')

            fcenter_ax = fig.add_subplot(gs[:, 1:-1])
            fcenter_ax.imshow(img, cmap ='gray')
            for (y, x) in idxs:
                # x = ((x // fact) + 0.5) * fact
                # y = ((y // fact) + 0.5) * fact
                fcenter_ax.add_patch(plt.Circle((x, y), fact // 2, color='r'))
                fcenter_ax.axis('off')

            fname = f'frame{index}_dec_catten.png'
            plt.savefig(join(path_to_dec_catten, fname))


def plot_attention_weight(attention_weight, index, save_dir: str = '/output/visual_attention_weight'):
    """
    Plot attention weight from decoder in each layer

    :param attention_weight: [num_layer, 1, downsample_tgt_hw, downsample_src_hw]
    :param save_dir: where to save the visual attention_weight_fig
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    for i, att_wt_layer in enumerate(attention_weight):
        att_wt_layer = torch2cv2(att_wt_layer/att_wt_layer.max())
        fname = f'frame{index}_layer{i}_attention_weight.png'
        cv2.imwrite(join(save_dir, fname), att_wt_layer)
        

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def optimal_crop_size(max_size, max_subsample_factor, safety_margin=0):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    crop_size += safety_margin * pow(2, max_subsample_factor)
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders, safety_margin=0):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        self.width_crop_size = optimal_crop_size(self.width, num_encoders, safety_margin)
        self.height_crop_size = optimal_crop_size(self.height, num_encoders, safety_margin)

        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ZeroPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)

    def crop(self, img):
        return img[..., self.iy0:self.iy1, self.ix0:self.ix1]


def format_power(size):
    power = 1e3
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]


def flow2bgr_np(disp_x, disp_y, max_magnitude=None):
    """
    Convert an optic flow tensor to an RGB color map for visualization
    Code adapted from: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py#L339

    :param disp_x: a [H x W] NumPy array containing the X displacement
    :param disp_y: a [H x W] NumPy array containing the Y displacement
    :returns bgr: a [H x W x 3] NumPy array containing a color-coded representation of the flow [0, 255]
    """
    assert(disp_x.shape == disp_y.shape)
    H, W = disp_x.shape

    # X, Y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))

    # flow_x = (X - disp_x) * float(W) / 2
    # flow_y = (Y - disp_y) * float(H) / 2
    # magnitude, angle = cv.cartToPolar(flow_x, flow_y)
    # magnitude, angle = cv.cartToPolar(disp_x, disp_y)

    # follow alex zhu color convention https://github.com/daniilidis-group/EV-FlowNet

    flows = np.stack((disp_x, disp_y), axis=2)
    magnitude = np.linalg.norm(flows, axis=2)

    angle = np.arctan2(disp_y, disp_x)
    angle += np.pi
    angle *= 180. / np.pi / 2.
    angle = angle.astype(np.uint8)

    if max_magnitude is None:
        v = np.zeros(magnitude.shape, dtype=np.uint8)
        cv.normalize(src=magnitude, dst=v, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    else:
        v = np.clip(255.0 * magnitude / max_magnitude, 0, 255)
        v = v.astype(np.uint8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = angle
    hsv[..., 2] = v
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    return bgr


def recursive_clone(tensor):
    """
    Assumes tensor is a torch.tensor with 'clone()' method, possibly
    inside nested iterable.
    E.g., tensor = [(pytorch_tensor, pytorch_tensor), ...]
    """
    if hasattr(tensor, 'clone'):
        return tensor.clone()
    try:
        return type(tensor)(recursive_clone(t) for t in tensor)
    except TypeError:
        print('{} is not iterable and has no clone() method.'.format(tensor))


def get_height_width(data_loader):
    for d in data_loader:
        return d['events'].shape[-2:]  # d['events'] is a ... x H x W voxel grid 


def torch2cv2(image):
    """convert torch tensor to format compatible with cv2.imwrite"""
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy() 
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def append_timestamp(path, description, timestamp):
    with open(path, 'a') as f:
        f.write('{} {:.15f}\n'.format(description, timestamp))


def setup_output_folder(output_folder):
    """
    Ensure existence of output_folder and overwrite output_folder/timestamps.txt file.
    Returns path to output_folder/timestamps.txt & output_folder/infer_loss.txt
    """
    ensure_dir(output_folder)
    print('Saving to: {}'.format(output_folder))
    timestamps_path = join(output_folder, 'timestamps.txt')
    compute_infer_loss_path = join(output_folder, 'infer_loss.txt')
    open(timestamps_path, 'w').close()  # overwrite with emptiness
    open(compute_infer_loss_path, 'w').close()
    return timestamps_path, compute_infer_loss_path

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
