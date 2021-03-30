#!/usr/bin/env python3
'''
Generate synthetic macular hole-like images.
'''
import os
import random

import numpy as np
import skimage.io

def generate_macular_holes(n_macular_holes, root_dir):
    for image_num in range(n_macular_holes):
        SHAPE = (49, 376, 321)

        # Base img, some random noise but dark
        img = np.random.rand(*SHAPE).astype(np.float64) * 0.3
        gt = np.zeros(SHAPE).astype(np.float64)

        #############################################################
        # Circle Arc creation
        #############################################################
        START_X = -10
        END_X = SHAPE[2] + 10
        circle_centre_x = (np.random.rand() * (END_X - START_X)) + START_X

        START_Y = SHAPE[1] + 40
        END_Y = SHAPE[1] + 80
        circle_centre_y = (np.random.rand() * (END_Y - START_Y)) + START_Y

        print(f'circle centre: ({circle_centre_x},{circle_centre_y})')

        image_centre_x = SHAPE[2] / 2.0
        image_centre_y = SHAPE[1] / 2.0

        print(f'image centre: ({image_centre_x},{image_centre_y})')

        arc_half_len = 80
        radius0 = np.sqrt(
                np.square((image_centre_x - circle_centre_x)) +
                np.square(((image_centre_y + arc_half_len) - circle_centre_y)))

        radius1 = np.sqrt(
                np.square((image_centre_x - circle_centre_x)) +
                np.square(((image_centre_y - arc_half_len) - circle_centre_y)))
        print(f'radius: {radius0}, {radius1}')


        rectangle_x1 = image_centre_x - 50
        rectangle_y1 = image_centre_y - 100

        rectangle_x2 = rectangle_x1 + 100
        rectangle_y2 = rectangle_y1 + 100

        ranges = [random.random(), random.random()]
        START_Z = 1
        END_Z = SHAPE[0] - 1
        range_start = START_Z + (min(ranges) * (END_Z - START_Z))
        range_end = START_Z + (max(ranges) * (END_Z - START_Z))

        print(f'range_start: {range_start} {range_end}')

        for slice_idx in range(SHAPE[0]):
            print(f'slice_idx : {slice_idx}')
            for i in range(SHAPE[2]):
                for j in range(SHAPE[1]):
                    x = i
                    y = j
                    dist = np.sqrt(
                        np.square((x - circle_centre_x)) +
                        np.square((y - circle_centre_y)))
                    #print(f'distance from ({x}, {y}) to circle centre is {dist}')
                    within_xy_arc = dist > radius0 and dist < radius1
                    within_z_macular_hole = (slice_idx > range_start and slice_idx < range_end)
                    within_xy_macular_hole = (x > rectangle_x1 and x < rectangle_x2 and y > rectangle_y1 and y < rectangle_y2)
                    if within_xy_arc and not (within_z_macular_hole and within_xy_macular_hole):
                        img[slice_idx, y, x] += 0.5
                    else:
                        if within_xy_arc and (within_z_macular_hole and within_xy_macular_hole):
                            gt[slice_idx, y, x] = 1.0

        #############################################################
        # End Circle Arc creation
        #############################################################

        #############################################################
        # Begin creation of macular hole
        #############################################################

        #############################################################
        # End creation of macular hole
        #############################################################

        img = (img * 255).astype(np.uint8)
        gt = (gt * 255).astype(np.uint8)
        skimage.io.imsave(f'{root_dir}/im/{image_num:03}.tif', img)
        skimage.io.imsave(f'{root_dir}/gt/{image_num:03}.tif', gt)


ROOT_DIR = '../dataset'
N_TRAIN = 35
N_VALIDATION = 10
N_TEST = 9

os.makedirs(f'{ROOT_DIR}/train/im', exist_ok=True)
os.makedirs(f'{ROOT_DIR}/train/gt', exist_ok=True)
os.makedirs(f'{ROOT_DIR}/validation/im', exist_ok=True)
os.makedirs(f'{ROOT_DIR}/validation/gt', exist_ok=True)
os.makedirs(f'{ROOT_DIR}/test/im', exist_ok=True)
os.makedirs(f'{ROOT_DIR}/test/gt', exist_ok=True)

generate_macular_holes(N_TRAIN, f'{ROOT_DIR}/train')
generate_macular_holes(N_VALIDATION, f'{ROOT_DIR}/validation')
generate_macular_holes(N_TEST, f'{ROOT_DIR}/test')
