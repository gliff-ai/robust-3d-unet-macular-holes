import os
import time

import numpy as np
import skimage.io
import torch

import common


def resize(tensor, shape):
    result = torch.nn.functional.interpolate(tensor,
        size=shape,
        mode='trilinear',
        align_corners=False)
    return result


def run(model, dataset_group, exp_id, weights_filename):
    device = common.setup_device()
    weights_folder = f'../out/cli-seg-weights/{exp_id}'
    weights_path = os.path.join(weights_folder, f'{weights_filename}.pth')
    if not os.path.exists(weights_path):
        raise ValueError('The weights folder does not exist')
    weights = torch.load(weights_path)
    model.to(device=device)
    model.load_state_dict(weights['model_state_dict'])

    for dataset in ['validation', 'test']:
        img_dir = os.path.join(f'../out/cli-seg-infer/{exp_id}/{dataset}/im')
        os.makedirs(img_dir, exist_ok=True)
        result_dir = os.path.join(f'../out/cli-seg-infer/{exp_id}/{dataset}/result')
        os.makedirs(result_dir, exist_ok=True)
        full_size_thresholded_result_dir = os.path.join(f'../out/cli-seg-infer/{exp_id}/{dataset}/full_size_thresholded_result')
        os.makedirs(full_size_thresholded_result_dir, exist_ok=True)

        with torch.no_grad():
            img_accum = []
            result_accum = []
            gt_accum = []
            full_size_gt_accum = []
            result_crops = []
            for img, gt, full_size_img, full_size_gt, img_name in getattr(dataset_group, dataset):
                assert(len(img_name) == 1)
                img = img.to(device=device)
                before = time.time()
                result = model(img)
                after = time.time()
                print(f'<exp_id={exp_id},time taken for inference: {after - before} seconds')
                if dataset_group.is_2d():
                    # For 2D, need to assemble all slices into one 3D image
                    img_accum.append(img[0, 0, 4])
                    result_accum.append(result[0, 0, 0])
                    gt_accum.append(gt[0, 0, 0])
                    full_size_gt_accum.append(full_size_gt[0, 0, 0])
                    if '48' not in img_name[0]:  # Last image in slices
                        continue
                    else:
                        img = torch.stack(img_accum)
                        img = torch.unsqueeze(img, 0)
                        img = torch.unsqueeze(img, 0)
                        result = torch.stack(result_accum)
                        result = torch.unsqueeze(result, 0)
                        result = torch.unsqueeze(result, 0)
                        gt = torch.stack(gt_accum)
                        gt = torch.unsqueeze(gt, 0)
                        gt = torch.unsqueeze(gt, 0)
                        full_size_gt = torch.stack(full_size_gt_accum)
                        full_size_gt = torch.unsqueeze(full_size_gt, 0)
                        full_size_gt = torch.unsqueeze(full_size_gt, 0)
                        img_accum = []
                        result_accum = []
                        gt_accum = []
                        full_size_gt_accum = []
                        img_name = [f"{'_'.join(img_name[0].split('_')[:-1])}.tif"]

                print(f'is_cropped: {getattr(dataset_group, "is_cropped", False)}')
                if getattr(dataset_group, 'is_cropped', False):
                    if '_8' in img_name[0]: # TODO: Make this more generic
                        result_crops.append(result.detach().cpu().numpy())
                        # Combine all results together into a correctly sized image
                        combined_result = np.zeros(full_size_gt.shape)
                        print(f'combined_result.shape: {combined_result.shape}')
                        print(f'result_crops: {len(result_crops)}')
                        for x_crop_idx in range(3):
                            for y_crop_idx in range(3):
                                result_part = result_crops.pop(0)
                                crop_y_extent = result_part.shape[3]
                                crop_x_extent = result_part.shape[4]

                                if y_crop_idx == 0:
                                    y_start = 0
                                elif y_crop_idx == 1:
                                    midway = (combined_result.shape[3] / 2)
                                    y_start = int(midway - (crop_y_extent / 2))
                                elif y_crop_idx == 2:
                                    y_start = combined_result.shape[3] - crop_y_extent

                                if x_crop_idx == 0:
                                    x_start = 0
                                elif x_crop_idx == 1:
                                    midway = (combined_result.shape[4] / 2)
                                    x_start = int(midway - (crop_x_extent / 2))
                                elif x_crop_idx == 2:
                                    x_start = combined_result.shape[4] - crop_x_extent

                                print(f'result_part.shape: {result_part.shape}')
                                combined_result[:,:,:, y_start:y_start+crop_y_extent, x_start:x_start+crop_x_extent] = result_part
                                full_size_result = torch.from_numpy(combined_result)
                        result_crops = []

                        full_size_result = full_size_result[0]
                        full_size_img = full_size_img[0]
                        print(f'outputting images... {img_dir}, {img_name}')
                        print(f'shape: {full_size_img.shape},{full_size_result.cpu().numpy().shape},{full_size_gt[0].shape}')

                        common.output_images(img_dir, full_size_thresholded_result_dir, result_dir, img_name, full_size_img, full_size_result)
                    else:
                        result_crops.append(result.detach().cpu().numpy())
                else:
                    print(f'outputting images... {img_dir}, {img_name}')

                    print(f'shape: {result.cpu().numpy().shape} -> {full_size_gt[0, 0].shape}')
                    full_size_result = resize(result.cpu(), full_size_gt[0, 0].shape)
                    common.output_images(img_dir, full_size_thresholded_result_dir, result_dir, img_name, img, full_size_result)
