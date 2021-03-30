import os

import numpy as np
import pathlib
import skimage.io
import torch

import datasets.dataset_2d_slices
import datasets.dataset_3d
import models.unet_3d_proposal
import models.unet_3d_residual_proposal
import models.unet_3d_slices_residual


OPTS = '''
  combine_probabilities
  dataset_root
  dataset_size_code
  dataset_type
  epochs
  iter_per_epoch
  learning_rate
  model_id
  weight_decay'''



def make_dataset_group(args):
    dataset_root = args.dataset_root
    if args.dataset_type == '3d':
        dataset_group = datasets.dataset_3d.Tiff3DDatasetGroup(dataset_root, args.dataset_size_code, batch_size=1, combine_probabilities=args.combine_probabilities)
    elif args.dataset_type == '2d_slices':
        dataset_group = datasets.dataset_2d_slices.Tiff3DTo2DSlicesDatasetGroup(dataset_root, args.dataset_size_code, combine_probabilities=args.combine_probabilities)
    else:
        raise NotImplementedError('Not a valid dataset: {arg.dataset_type}')
    return dataset_group


def make_model(model_id):
    if model_id == 'unet_3d_proposal':
        model = models.unet_3d_proposal.UNet3DProposal()
    elif model_id == 'unet_3d_residual_proposal':
        model = models.unet_3d_residual_proposal.UNet3DResidualProposal()
    elif model_id == 'unet_3d_slices_residual':
        model = models.unet_3d_slices_residual.UNet3DSlicesResidual()
    else:
        raise NotImplementedError('Not a valid model')
    return model


def normalize(arr):
    denominator = arr.max() - arr.min()
    if denominator == 0:
        return arr
    return (arr - arr.min()) / denominator


def setup_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('using CUDA device')
        device = torch.device('cuda')
    else:
        print('using CPU device')
    return device


def output_images(img_dir, thresholded_result_dir, result_dir, img_names, imgs, results, suffix=None):
    batch_size = len(img_names)
    print(f'batch: {batch_size}')
    for i in range(batch_size):
        img_name = img_names[i]
        img = (imgs[i] * 255).cpu().numpy().astype(np.uint8)
        result = (results[i] * 255).cpu().numpy().astype(np.uint8)
        print(f'img: {img.min()} {img.max()}')
        print(f'result: {result.min()} {result.max()}')
        thresholded_result = ((results[i] > 0.5) * 255).cpu().numpy().astype(np.uint8)
        print(f'thresholded_result: {thresholded_result.min()} {thresholded_result.max()}')
        if thresholded_result.sum() > 0:
            print(f'output_images img_name {img_name}')
            print(f'output_images img has shape {img.shape}')
            print(f'output_images result has shape {result.shape}')
            print(f'output_images thresholded_result has shape {thresholded_result.shape}')
            if img.shape[0] == 1:
                img = img[0]
                result = result[0]
                thresholded_result = thresholded_result[0]
            else:
                pass
            if suffix is not None:
                img_name = f'{img_name.replace(".tif", "")}_{suffix}.tif'
            #print(f'output_images tranimg has shape {img.shape}')
            #print(f'output_images tranresult has shape {result.shape}')
            print(f'outputting: {os.path.join(img_dir, img_name)}...')
            print(f'output_images img_name {img_name}')
            print(f'output_images img has shape {img.shape}')
            print(f'output_images result has shape {result.shape}')
            print(f'output_images thresholded_result has shape {thresholded_result.shape}')
            if not img_name.endswith('.tif'):
                img_name = f'{img_name}.tif'
            skimage.io.imsave(os.path.join(img_dir, img_name), img)
            skimage.io.imsave(os.path.join(result_dir, img_name), result)
            skimage.io.imsave(os.path.join(thresholded_result_dir, img_name), thresholded_result)
