import os

import skimage.io
import torch
import torch.utils
import torch.utils.data

import common
import datasets.cycle


# Z, Y, X
IM_SIZE = {
    'highres3d':   ( 49, 188, 160)
}

GT_SIZE = {
    'highres3d':   ( 49, 188, 160),
}


class Tiff3DDatasetGroup:
    def __init__(self, root_dir, dataset_size_code='cicek', batch_size=1, combine_probabilities=False):
        self.training_dataset = Tiff3DDataset(root_dir, 'train', dataset_size_code, combine_probabilities)
        self.training = datasets.cycle.Cycle(torch.utils.data.DataLoader(self.training_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=False))

        self.validation_dataset = Tiff3DDataset(root_dir, 'validation', dataset_size_code, False)
        self.validation = torch.utils.data.DataLoader(self.validation_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=False)

        self.test_dataset = Tiff3DDataset(root_dir, 'test', dataset_size_code, False)
        self.test = torch.utils.data.DataLoader(self.test_dataset,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=False)

        root_dir = os.path.join(root_dir, 'train')
        im_dir = os.path.join(root_dir, 'im')
        filename_list = sorted(list(os.listdir(im_dir)))
        first_im_file = skimage.img_as_ubyte(
            skimage.io.imread(
                os.path.join(im_dir, filename_list[0])))
        #self.first_im_shape = [first_im_file.shape[2], first_im_file.shape[1], first_im_file.shape[0]]
        self.first_im_shape = first_im_file.shape

    def __repr__(self):
        return f'Tiff3DDatasetGroup(training={self.training_dataset},validation={self.validation_dataset},test={self.test_dataset})'

    def is_2d(self):
        return False

    def resize_to_full_size(self, tensor):
        return torch.nn.functional.interpolate(tensor,
            size=self.first_im_shape,
            mode='trilinear',
            align_corners=False)


class Tiff3DDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, dataset_name, dataset_size_code, combine_probabilities):
        super(Tiff3DDataset, self).__init__()
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.dataset_size_code = dataset_size_code
        root_dir = os.path.join(root_dir, dataset_name)
        self.im_root_dir = os.path.join(root_dir, 'im')
        self.gt_root_dir = os.path.join(root_dir, 'gt')
        self.filename_list = sorted(list(os.listdir(self.gt_root_dir)))
        self.dataset_size_code = dataset_size_code
        self.combine_probabilities = combine_probabilities
        if self.combine_probabilities:
            self.unique_filename_list = sorted(list(set([i.split('_')[0] for i in self.filename_list])))

    def __repr__(self):
        return f'Tiff3DDataset(root_dir={self.root_dir},dataset_name={self.dataset_name},dataset_size_code={self.dataset_size_code})'

    def __getitem__(self, idx):
        im_file_idx = idx
        if self.combine_probabilities:
            self.unique_filename_list = sorted(list(set([i.split('_')[0] for i in self.filename_list])))
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
        im = torch.unsqueeze(im, 0)
        full_size_im = im
        im = torch.unsqueeze(im, 0)
        im = torch.nn.functional.interpolate(im,
            size=IM_SIZE[self.dataset_size_code],
            mode='trilinear',
            align_corners=False)
        im = im[0]

        if self.combine_probabilities:
            filtered_gt_files = sorted([f for f in self.filename_list if im_name in f])
            combined_gt = None
            full_size_gt = None
            for gt_file in filtered_gt_files:
                gt_path = os.path.join(self.gt_root_dir, gt_file)
                gt = skimage.img_as_ubyte(skimage.io.imread(gt_path))
                gt = common.normalize(gt) / len(filtered_gt_files)
                gt = torch.from_numpy(gt).float()
                gt = torch.unsqueeze(gt, 0)
                if full_size_gt is None:
                    full_size_gt = gt
                else:
                    full_size_gt += gt
                gt = torch.unsqueeze(gt, 0)
                gt = torch.nn.functional.interpolate(gt,
                    size=GT_SIZE[self.dataset_size_code],
                    mode='trilinear',
                    align_corners=False)
                gt = gt[0]
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
            gt = torch.unsqueeze(gt, 0)
            full_size_gt = gt
            gt = torch.unsqueeze(gt, 0)
            gt = torch.nn.functional.interpolate(gt,
                size=GT_SIZE[self.dataset_size_code],
                mode='trilinear',
                align_corners=False)
            gt = gt[0]

        return im, gt, full_size_im, full_size_gt, im_name

    def __len__(self):
        if self.combine_probabilities:
            return len(self.unique_filename_list)
        else:
            return len(self.filename_list)
