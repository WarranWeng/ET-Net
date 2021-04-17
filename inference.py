import argparse
import torch
from os.path import join
import cv2
from tqdm import tqdm

from model import model as model_arch
from data_loader.data_loaders import InferenceDataLoader
from model.model import ColorNet
from utils.util import CropParameters, get_height_width, torch2cv2, \
                       append_timestamp, setup_output_folder, minmax_normalization
from utils.timers import CudaTimer
from utils.henri_compatible import make_henri_compatible

from parse_config import ConfigParser
from utils.compute_infer_loss import compute_infer_loss


def legacy_compatibility(args, checkpoint):
    assert not (args.e2vid and args.firenet_legacy)
    if args.e2vid:
        args.legacy_norm = True
        final_activation = 'sigmoid'
    elif args.firenet_legacy:
        args.legacy_norm = True
        final_activation = ''
    # Make compatible with Henri saved models
    if not isinstance(checkpoint.get('config', None), ConfigParser) or args.e2vid or args.firenet_legacy:
        checkpoint = make_henri_compatible(checkpoint, final_activation)
    if args.firenet_legacy:
        checkpoint['config']['arch']['type'] = 'FireNet_legacy'
    return args, checkpoint


def load_model(checkpoint, device):
    config = checkpoint['config']
    state_dict = checkpoint['state_dict']

    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', model_arch)
    logger.info(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    if args.color:
        model = ColorNet(model)
    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def main(args, model, device):
    dataset_kwargs = {'transforms': {},
                      'max_length': None,
                      'sensor_resolution': None,
                      'num_bins': 5,
                      'filter_hot_events': args.filter_hot_events,
                      'voxel_method': {'method': args.voxel_method,
                                       'k': args.k,
                                       't': args.t,
                                       'sliding_window_w': args.sliding_window_w,
                                       'sliding_window_t': args.sliding_window_t}
                      }

    if args.legacy_norm:
        print('Using legacy voxel normalization')
        dataset_kwargs['transforms'] = {'LegacyNorm': {}}

    data_loader = InferenceDataLoader(args.events_file_path, dataset_kwargs=dataset_kwargs, ltype=args.loader_type)

    height, width = get_height_width(data_loader)

    crop = CropParameters(width, height, int(args.num_encoder)) # num of encoder

    ts_fname, infer_loss_fname = setup_output_folder(args.output_folder)

    infer_loss = compute_infer_loss(infer_loss_fname, LPIPS_net_type='alex')

    model.reset_states()
    for i, item in enumerate(tqdm(data_loader)):

        # if i > 10:
        #     break
        voxel = item['events'].to(device)
        img = item['frame'].to(device)

        if not args.color:
            voxel = crop.pad(voxel)
        with CudaTimer('Inference'):
            output = model(voxel)

        if args.use_minmax_morm:
            output['image'] = minmax_normalization(output['image'], output['image'].device)
            
        infer_loss(crop.crop(output['image']), img)

        # save sample images, or do something with output here
        if args.color:
            image = output['image']
        else:
            image = crop.crop(output['image'])
            image = torch2cv2(image)
        fname = 'frame_{:010d}.png'.format(i)
        cv2.imwrite(join(args.output_folder, fname), image)

        append_timestamp(ts_fname, fname, item['timestamp'].item())

    print(infer_loss.write_loss())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--checkpoint_path', required=True, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--events_file_path', required=True, type=str,
                        help='path to events (HDF5)')
    parser.add_argument('--output_folder', default="result", type=str,
                        help='where to save outputs to')
    parser.add_argument('--device', default='0', type=str,
                        help='indices of GPUs to enable')
    parser.add_argument('--color', action='store_true', default=False,
                        help='Perform color reconstruction')
    parser.add_argument('--voxel_method', default='between_frames', type=str,
                        help='which method should be used to form the voxels',
                        choices=['between_frames', 'k_events', 't_seconds'])
    parser.add_argument('--k', type=int,
                        help='new voxels are formed every k events (required if voxel_method is k_events)')
    parser.add_argument('--sliding_window_w', type=int,
                        help='sliding_window size (required if voxel_method is k_events)')
    parser.add_argument('--t', type=float,
                        help='new voxels are formed every t seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--sliding_window_t', type=float,
                        help='sliding_window size in seconds (required if voxel_method is t_seconds)')
    parser.add_argument('--loader_type', default='H5', type=str,
                        help='Which data format to load (HDF5 recommended)')
    parser.add_argument('--filter_hot_events', action='store_true',
                        help='If true, auto-detect and remove hot pixels')
    parser.add_argument('--legacy_norm', action='store_true', default=False,
                        help='Normalize nonzero entries in voxel to have mean=0, std=1 according to Rebecq20PAMI and Scheerlinck20WACV.'
                        'If --e2vid or --firenet_legacy are set, --legacy_norm will be set to True (default False).')
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='set required parameters to run original e2vid as described in Rebecq20PAMI')
    parser.add_argument('--firenet_legacy', action='store_true', default=False,
                        help='set required parameters to run legacy firenet as described in Scheerlinck20WACV (not for retrained models using updated code)')
    parser.add_argument('--num_encoder', default=3, type=int)
    parser.add_argument('--use_minmax_morm', action='store_true', default=False,
                        help='If true, use minmax normalization')
    args = parser.parse_args()

    device = torch.device(f'cuda: {args.device}' if torch.cuda.is_available() else 'cpu')

    print('Loading checkpoint: {} ...'.format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))  ####### load to cpu
    args, checkpoint = legacy_compatibility(args, checkpoint)
    model = load_model(checkpoint, device)
    main(args, model, device)
