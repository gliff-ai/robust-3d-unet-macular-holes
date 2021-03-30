import os

import numpy as np
import skimage.io
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms
import torchvision.transforms.functional as tf

import common
import datasets.cycle


IM_SIZE = {
    'nocontext': 0,
    'macular': 4
}
GT_SIZE = {
    'nocontext': 0,
    'macular': 4
}


class Tiff3DTo2DSlicesDatasetGroup:
    def __init__(self, root_dir, dataset_size_code='macular', combine_probabilities=False):
        self.training_dataset = Tiff3DTo2DDatasetSlices(root_dir, 'train', dataset_size_code, combine_probabilities=combine_probabilities)
        self.training = datasets.cycle.Cycle(torch.utils.data.DataLoader(self.training_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=True,
                pin_memory=True))

        self.validation_dataset = Tiff3DTo2DDatasetSlices(root_dir, 'validation', dataset_size_code)
        self.validation = torch.utils.data.DataLoader(self.validation_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=True)

        self.test_dataset = Tiff3DTo2DDatasetSlices(root_dir, 'test', dataset_size_code)
        self.test = torch.utils.data.DataLoader(self.test_dataset,
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=True)

        root_dir = os.path.join(root_dir, 'train')
        im_dir = os.path.join(root_dir, 'im')
        filename_list = sorted(list(os.listdir(im_dir)))
        first_im_file = skimage.img_as_ubyte(
            skimage.io.imread(
                os.path.join(im_dir, filename_list[0])))
        self.first_im_shape = first_im_file.shape[1:]

    def is_2d(self):
        return True

    def __repr__(self):
        return f'Tiff3DTo2DDatasetGroup(training={self.training_dataset},validation={self.validation_dataset},test={self.test_dataset})'

    def resize_to_full_size(self, tensor):
        return tensor


class Tiff3DTo2DDatasetSlices(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset_name, dataset_size_code, combine_probabilities=False):
        super(Tiff3DTo2DDatasetSlices, self).__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.dataset_size_code = dataset_size_code
        root_dir = os.path.join(root_dir, dataset_name)
        self.im_root_dir = os.path.join(root_dir, 'im')
        self.gt_root_dir = os.path.join(root_dir, 'gt')
        self.filename_list = sorted(list(os.listdir(self.gt_root_dir)))
        self.dataset_size_code = dataset_size_code
        first_im_file = skimage.img_as_ubyte(
            skimage.io.imread(
                os.path.join(self.im_root_dir, self.filename_list[0])))
        self.num_slices_per_file = first_im_file.shape[0]

        self.combine_probabilities = combine_probabilities
        if self.combine_probabilities:
            self.unique_filename_list = sorted(list(set([i.split('_')[0] for i in self.filename_list])))

    def __repr__(self):
        return f'Tiff3DTo2DDatasetSlices(root_dir={self.root_dir},dataset_name={self.dataset_name},dataset_size_code={self.dataset_size_code},self.combine_probabilities={self.combine_probabilities})'

    def __getitem__(self, idx):
        im_file_idx = idx // self.num_slices_per_file
        slice_idx_within_file = idx % self.num_slices_per_file
        if self.combine_probabilities:
            im_name = self.unique_filename_list[im_file_idx]
            filtered_filenames = sorted([f for f in self.filename_list if im_name in f])
            full_name = filtered_filenames[0]
            im_path = os.path.join(self.im_root_dir, full_name)
        else:
            im_name = self.filename_list[im_file_idx]
            im_path = os.path.join(self.im_root_dir, im_name)

        im = skimage.img_as_ubyte(skimage.io.imread(im_path))

        im = common.normalize(im)

        im = torch.from_numpy(im).float()

        if self.combine_probabilities:
            filtered_gt_files = sorted([f for f in self.filename_list if im_name in f])
            combined_gt = None
            full_size_gt = None
            for gt_file in filtered_gt_files:
                gt_path = os.path.join(self.gt_root_dir, gt_file)
                gt = skimage.img_as_ubyte(skimage.io.imread(gt_path))
                gt = common.normalize(gt) / len(filtered_gt_files)
                gt = torch.from_numpy(gt).float()
                if full_size_gt is None:
                    full_size_gt = gt
                else:
                    full_size_gt += gt
                if combined_gt is None:
                    combined_gt = gt
                else:
                    combined_gt += gt
            gt = combined_gt
        else:
            gt_path = os.path.join(self.gt_root_dir, im_name)
            gt = skimage.img_as_ubyte(skimage.io.imread(gt_path))
            gt = common.normalize(gt)
            gt = torch.from_numpy(gt).float()

        # Do reflection of slices if at either end
        if slice_idx_within_file < IM_SIZE[self.dataset_size_code]:
            new_im = torch.zeros(((IM_SIZE[self.dataset_size_code]*2+1), im.shape[1], im.shape[2]))  # dim is context slices on either side and 1 is actual slice we are trying to segment
            n_reflected = IM_SIZE[self.dataset_size_code] - slice_idx_within_file
            for i in range(n_reflected):
                new_im[i, :, :] = im[n_reflected-i, :, :]
            new_im[n_reflected:new_im.shape[0]] = im[0:new_im.shape[0]-n_reflected]
            im = new_im
        elif slice_idx_within_file > (im.shape[0] - IM_SIZE[self.dataset_size_code] - 1):
            new_im = torch.zeros(((IM_SIZE[self.dataset_size_code]*2+1), im.shape[1], im.shape[2]))  # dim is context slices on either side and 1 is actual slice we are trying to segment
            n_reflected = (IM_SIZE[self.dataset_size_code] + 1) - (im.shape[0] - slice_idx_within_file)
            new_im[0:new_im.shape[0]-n_reflected] = im[slice_idx_within_file-IM_SIZE[self.dataset_size_code]:im.shape[0]]
            for i in range(n_reflected):
                new_im[new_im.shape[0]-n_reflected + i, :, :] = im[im.shape[0] - i - 2, :, :]
            im = new_im
        else:
            im = im[slice_idx_within_file-IM_SIZE[self.dataset_size_code]:slice_idx_within_file+IM_SIZE[self.dataset_size_code]+1, :, :]
        gt = gt[slice_idx_within_file, :, :]

        im = torch.unsqueeze(im, 0)
        gt = torch.unsqueeze(gt, 0)
        gt = torch.unsqueeze(gt, 0)

        return im, gt, im, gt, f'{im_name.replace(".tif", "")}_{slice_idx_within_file:03}.tif'

    def __len__(self):
        if self.combine_probabilities:
            return len(self.unique_filename_list) * self.num_slices_per_file
        else:
            return len(self.filename_list) * self.num_slices_per_file
