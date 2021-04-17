import os
import glob
from pathlib import Path


def mean(l):
    return 0 if len(l) == 0 else sum(l) / len(l)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def infer_model(pretrained_model_path, name, dataset_path='/path/to/dataset', is_append_for_metric=False,
                path_to_output='/path/to/output', num_encoder=3, use_minmax_norm=False, is_e2vid=False, is_firenet=False):
    if os.path.isdir(dataset_path):
        data_paths = sorted(glob.glob(os.path.join(dataset_path, '*.h5')))
    else:
        data_paths = [dataset_path]
    model_path = pretrained_model_path
    output_dir = os.path.join(path_to_output, f'{name}')
    metric_result_fname = os.path.join(output_dir, 'metric_result.txt')

    ensure_dir(output_dir)
    if not is_append_for_metric:
        open(metric_result_fname, 'w').close()

    result_of_all_sequences = {'mean_lpips':[], 'mean_mse':[], 'mean_ssim':[]}

    for data_path in data_paths:
        output_path = os.path.join(output_dir, data_path.split('/')[-1].split('.')[0])
        if use_minmax_norm:
            if is_e2vid:
                cmd = f'python inference.py --num_encoder {num_encoder} --checkpoint_path {model_path} \
                        --events_file_path {data_path} --output_folder {output_path} --use_minmax_morm --e2vid' 
            elif is_firenet:
                cmd = f'python inference.py --num_encoder {num_encoder} --checkpoint_path {model_path} \
                        --events_file_path {data_path} --output_folder {output_path} --use_minmax_morm --firenet_legacy' 
            else:
                cmd = f'python inference.py --num_encoder {num_encoder} --checkpoint_path {model_path} \
                        --events_file_path {data_path} --output_folder {output_path} --use_minmax_morm' 
        else:
            if is_e2vid:
                cmd = f'python inference.py --num_encoder {num_encoder} --checkpoint_path {model_path} \
                        --events_file_path {data_path} --output_folder {output_path} --e2vid' 
            elif is_firenet:
                cmd = f'python inference.py --num_encoder {num_encoder} --checkpoint_path {model_path} \
                        --events_file_path {data_path} --output_folder {output_path} --firenet_legacy' 
            else:
                cmd = f'python inference.py --num_encoder {num_encoder} --checkpoint_path {model_path} \
                        --events_file_path {data_path} --output_folder {output_path}' 
        info = os.popen(cmd).read()
        print(info)
        current_result = eval(info.split('\n')[-4])

        result_of_all_sequences['mean_lpips'].append(current_result['mean_lpips'])
        result_of_all_sequences['mean_mse'].append(current_result['mean_mse'])
        result_of_all_sequences['mean_ssim'].append(current_result['mean_ssim'])

    result_of_all_sequences['mean_lpips_for_all_sequences'] = mean(result_of_all_sequences['mean_lpips'])
    result_of_all_sequences['mean_mse_for_all_sequences'] = mean(result_of_all_sequences['mean_mse'])
    result_of_all_sequences['mean_ssim_for_all_sequences'] = mean(result_of_all_sequences['mean_ssim'])

    with open(metric_result_fname, 'w') as f:
        f.write('mean lpips for each sequence in this dataset: {}\n'.format(result_of_all_sequences['mean_lpips']))
        f.write('mean mse for each sequence in this dataset: {}\n'.format(result_of_all_sequences['mean_mse']))
        f.write('mean ssim for each sequence in this dataset: {}\n'.format(result_of_all_sequences['mean_ssim']))
        f.write('******************************\n')
        f.write('mean lpips for this dataset: {}\n'.format(result_of_all_sequences['mean_lpips_for_all_sequences']))
        f.write('mean mse for this dataset: {}\n'.format(result_of_all_sequences['mean_mse_for_all_sequences']))
        f.write('mean ssim for this dataset: {}\n'.format(result_of_all_sequences['mean_ssim_for_all_sequences']))

    return result_of_all_sequences


if __name__ == "__main__":
    infer_model('/path/to/model.pth', 'test',
                 dataset_path='/path/to/dataset', path_to_output='/path/to/output')


