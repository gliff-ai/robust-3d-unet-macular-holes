import csv
import datetime
import json
import logging
import os
import subprocess
import time

import numpy as np
import torch
import torch.optim

import common


logger = None


def get_git_hash():
    process = subprocess.Popen(['git', 'rev-parse', 'HEAD'], shell=False, stdout=subprocess.PIPE)
    return process.communicate()[0].strip().decode("utf-8")


def jaccard(result, gt):
    thresholded_result = (result > 0.5)
    thresholded_gt = (gt > 0.5)
    union = thresholded_result | thresholded_gt
    union_sum = union.sum()
    if union_sum == 0:
        # If both are zero, then Jaccard is undefined but I think that counts as a perfect result
        return 1.0
    else:
        intersection = thresholded_result & thresholded_gt
        return (intersection.sum() / float(union_sum))



def normal_training_loop(dataset_group, criterion, model, optimizer, iter_per_epoch):
    train_losses = np.zeros(iter_per_epoch)
    train_jaccards = np.zeros(iter_per_epoch)
    train_jaccards_full_size = np.zeros(iter_per_epoch)

    load_start = time.time()
    i = 0
    while i < iter_per_epoch:
        img, gt, full_size_img, full_size_gt, img_name = dataset_group.training.next()
        logger.debug(f'running training on img <{img_name}>')

        img = img.to(device=DEVICE)
        gt = gt.to(device=DEVICE)
        full_size_gt = full_size_gt.to(device=DEVICE)
        load_finish = time.time()
        logger.info(f'perf,load_finished,{load_finish - load_start}')
        start = time.time()
        optimizer.zero_grad()

        result = model(img)
        loss = criterion(result, gt)
        loss.backward()
        optimizer.step()

        finish = time.time()
        logger.info(f'perf,model,{finish - start}')
        train_losses[i] = loss.item()
        full_size_result = dataset_group.resize_to_full_size(result)
        start = time.time()
        train_jaccards[i] = jaccard(result, gt)
        train_jaccards_full_size[i] = jaccard(full_size_result, full_size_gt)
        finish = time.time()
        logger.info(f'perf,jaccard,{finish - start}')
        i += 1
    return train_losses, train_jaccards, train_jaccards_full_size


def save(model, optimizer, checkpoint_folder, weights_filename):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               os.path.join(checkpoint_folder, f'{weights_filename}.pth'))


def setup_logging(exp_id):
    log_filename = f'../out/cli-seg-logs/{exp_id}.log'
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%dT%H:%M:%SZ')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def run(model, dataset_group, epochs, learning_rate, exp_id, iter_per_epoch, weight_decay):
    total_start = time.time()
    global logger
    logger = setup_logging(exp_id)
    global DEVICE
    DEVICE = common.setup_device()
    git_hash = get_git_hash()
    logger.info(f'Running on git revision: <{git_hash}> with params: model={type(model)},dataset_group={dataset_group}, epochs={epochs}, learning_rate={learning_rate}, exp_id={exp_id}, weight_decay={weight_decay}')
    model = model.to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    csv_filename = f'../out/cli-seg-results/cli_train_stats_{exp_id}.csv'
    detailed_csv_filename = f'../out/cli-seg-results/cli_train_stats_detailed_{exp_id}.csv'
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # Defaults
    best_validation_jaccard = 0.0
    best_test_jaccard = 0.0
    best_test_jaccard_full_size = 0.0
    start_epoch = 0
    csv_file_mode = 'w'

    criterion = torch.nn.BCEWithLogitsLoss()

    # Resume from previous run if checkpoint available
    weights_filename = 'checkpoint'
    checkpoint_folder = f'../out/cli-seg-weights/{exp_id}'
    if os.path.exists(checkpoint_folder):
        csv_file_mode = 'a'
        logger.info(f'Resuming weights from folder <{checkpoint_folder}>')
        with open(os.path.join(checkpoint_folder, f'{weights_filename}_metadata.json'), 'r') as checkpoint_info_file:
            metadata = json.load(checkpoint_info_file)

            start_epoch = metadata['epoch'] + 1
            best_validation_jaccard = metadata['best_validation_jaccard']
            if 'best_test_jaccard' in metadata:
                best_test_jaccard = metadata['best_test_jaccard']
            if 'best_test_jaccard_full_size' in metadata:
                best_test_jaccard_full_size = metadata['best_test_jaccard_full_size']
            checkpoint = torch.load(os.path.join(checkpoint_folder, f'{weights_filename}.pth'))
            logger.info(f'best_validation_jaccard <{best_validation_jaccard}>')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    logger.info('###############################################################')
    logger.info(f'# Beginning new run from epoch {start_epoch} to epoch {start_epoch + epochs - 1}')
    logger.info('###############################################################')

    logger.info(f'open({csv_filename}, mode={csv_file_mode}...')
    with open(csv_filename, mode=csv_file_mode, newline='') as csvfile:
        with open(detailed_csv_filename, mode=csv_file_mode, newline='') as detailed_csvfile:
            csv_writer = csv.writer(csvfile)
            detailed_csv_writer = csv.writer(detailed_csvfile)
            if start_epoch == 0:
                csv_writer.writerow(['epoch',
                    'train_loss', 'train_jaccard', 'train_jaccard_90', 'train_jaccard_std', 'train_jaccard_full_size', 'train_jaccard_90', 'train_jaccard_full_size_std',
                    'validation_loss', 'validation_jaccard', 'validation_jaccard_90', 'validation_jaccard_std', 'validation_jaccard_full_size', 'validation_jaccard_90', 'validation_jaccard_full_size_std',
                    'test_loss', 'test_jaccard', 'test_jaccard_90', 'test_jaccard_std', 'test_jaccard_full_size', 'test_jaccard_90', 'test_jaccard_full_size_std'])

                detailed_csv_writer.writerow(['epoch', 'dataset', 'im_name', 'jaccard', 'jaccard_full_size'])
            for epoch in range(start_epoch, start_epoch + epochs):
                logger.info('###############################################################')
                logger.info(f'# Started epoch {epoch}')
                logger.info('###############################################################')
                #######################################################################
                # Training
                #######################################################################
                logger.info('###############################################################')
                logger.info('# Started training set evaluation')
                logger.info('###############################################################')
                start = time.time()
                train_losses, train_jaccards, train_jaccards_full_size = normal_training_loop(dataset_group, criterion, model, optimizer, iter_per_epoch)

                finish = time.time()
                logger.info(f'perf,train,{finish - start}')
                logger.info('###############################################################')
                logger.info(f'# Finished training set loop with jaccard: {train_jaccards.mean()})')
                logger.info('###############################################################')

                #######################################################################
                # Validation
                #######################################################################
                logger.info('###############################################################')
                logger.info('# Started validation set evaluation')
                logger.info('###############################################################')
                start = time.time()
                len_validation = len(dataset_group.validation)
                if dataset_group.is_2d():
                    len_validation //= 49
                validation_losses = np.zeros(len_validation)
                validation_jaccards = np.zeros(len_validation)
                validation_jaccards_full_size = np.zeros(len_validation)

                i = 0
                with torch.no_grad():
                    img_accum = []
                    result_accum = []
                    gt_accum = []
                    full_size_gt_accum = []
                    for img, gt, full_size_img, full_size_gt, img_name in dataset_group.validation:
                        assert(len(img_name) == 1)
                        img = img.to(device=DEVICE)
                        gt = gt.to(device=DEVICE)
                        full_size_gt = full_size_gt.to(device=DEVICE)
                        result = model(img)

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

                        loss = criterion(result, gt)
                        validation_losses[i] = loss.item()
                        full_size_result = dataset_group.resize_to_full_size(result)
                        validation_jaccards[i] = jaccard(result, gt)
                        validation_jaccards_full_size[i] = jaccard(full_size_result, full_size_gt)

                        assert len(img_name) == 1, "This code can only handle batch size == 1"
                        detailed_csv_writer.writerow([epoch, 'validation', img_name[0], validation_jaccards[i], validation_jaccards_full_size[i]])
                        i += 1
                        # End is_cropped Hack

                        
                    # For memory
                    del img
                    del gt
                    del full_size_gt
                    del result
                    del loss
                    del full_size_result

                finish = time.time()
                logger.info(f'perf,validation,{finish - start}')
                logger.info('###############################################################')
                logger.info(f'# Finished validation set evaluation with jaccard: {validation_jaccards.mean()})')
                logger.info('###############################################################')

                #######################################################################
                # Test
                #######################################################################
                logger.info('###############################################################')
                logger.info('# Started test set evaluation')
                logger.info('###############################################################')
                start = time.time()

                len_test = len(dataset_group.test)
                if dataset_group.is_2d():
                    len_test //= 49
                test_losses = np.zeros(len_test)
                test_jaccards = np.zeros(len_test)
                test_jaccards_full_size = np.zeros(len_test)
                i = 0
                with torch.no_grad():
                    for img, gt, full_size_img, full_size_gt, img_name in dataset_group.test:
                        img = img.to(device=DEVICE)
                        gt = gt.to(device=DEVICE)
                        full_size_gt = full_size_gt.to(device=DEVICE)
                        result = model(img)
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

                    
                        loss = criterion(result, gt)
                        test_losses[i] = loss.item()
                        full_size_result = dataset_group.resize_to_full_size(result)

                        test_jaccards[i] = jaccard(result, gt)
                        test_jaccards_full_size[i] = jaccard(full_size_result, full_size_gt)

                        assert len(img_name) == 1, "This code can only handle batch size == 1"
                        detailed_csv_writer.writerow([epoch, 'test', img_name[0], test_jaccards[i], test_jaccards_full_size[i]])
                        i += 1

                    # For memory
                    del img
                    del gt
                    del full_size_gt
                    del result
                    del loss
                    del full_size_result

                finish = time.time()
                logger.info(f'perf,test,{finish - start}')
                logger.info('###############################################################')
                logger.info(f'# Finished test set evaluation with jaccard: {test_jaccards.mean()})')
                logger.info('###############################################################')
                csv_line = [epoch,
                    train_losses.mean(), train_jaccards.mean(), np.percentile(train_jaccards, 90),  train_jaccards.std(), train_jaccards_full_size.mean(), np.percentile(train_jaccards_full_size, 90),  train_jaccards_full_size.std(),
                    validation_losses.mean(), validation_jaccards.mean(), np.percentile(validation_jaccards, 90),  validation_jaccards.std(), validation_jaccards_full_size.mean(), np.percentile(validation_jaccards_full_size, 90),  validation_jaccards_full_size.std(),
                    test_losses.mean(), test_jaccards.mean(), np.percentile(test_jaccards, 90),  test_jaccards.std(), test_jaccards_full_size.mean(), np.percentile(test_jaccards_full_size, 90),  test_jaccards_full_size.std()]
                logger.info(f'writing to csv: {csv_line}')
                csv_writer.writerow(csv_line)
                detailed_csvfile.flush()
                csvfile.flush()

                if validation_jaccards.mean() > best_validation_jaccard:
                    best_validation_jaccard = validation_jaccards.mean()
                    #######################################################################
                    # Output at the end of each epoch where validation accuracy is increasing
                    #######################################################################
                    logger.info('# This step has validation jaccard better than or equal to the previous best, outputting weights...')
                    os.makedirs(checkpoint_folder, exist_ok=True)
                    weights_filename = 'best_validation'
                    save(model, optimizer, checkpoint_folder, weights_filename)
                    logger.info(f"open(os.path.join({checkpoint_folder}, '{weights_filename}_metadata.json'), 'w'")
                    with open(os.path.join(checkpoint_folder, f'{weights_filename}_metadata.json'), 'w') as checkpoint_info_file:
                        json_dict = dict(epoch=epoch,
                                       exp_id=exp_id,
                                       train_jaccard=float(train_jaccards.mean()),
                                       validation_jaccard=float(validation_jaccards.mean()),
                                       test_jaccard=float(test_jaccards.mean()),
                                       best_validation_jaccard=float(best_validation_jaccard),
                                       timestamp=f"{datetime.datetime.now().isoformat()}")
                        logger.info(f"Outputting to weights file {weights_filename} metadata: {json.dumps(json_dict)}")
                        json.dump(json_dict,
                                  checkpoint_info_file)

                if test_jaccards.mean() > best_test_jaccard:
                    best_test_jaccard = test_jaccards.mean()
                    #######################################################################
                    # Output at the end of each epoch where test accuracy is increasing
                    #######################################################################
                    logger.info('# This step has test jaccard better than or equal to the previous best, outputting weights...')
                    os.makedirs(checkpoint_folder, exist_ok=True)
                    weights_filename = 'best_test'
                    save(model, optimizer, checkpoint_folder, weights_filename)

                    logger.info(f"open(os.path.join({checkpoint_folder}, '{weights_filename}_metadata.json'), 'w'")
                    with open(os.path.join(checkpoint_folder, f'{weights_filename}_metadata.json'), 'w') as checkpoint_info_file:
                        json_dict = dict(epoch=epoch,
                                       exp_id=exp_id,
                                       train_jaccard=float(train_jaccards.mean()),
                                       validation_jaccard=float(validation_jaccards.mean()),
                                       test_jaccard=float(test_jaccards.mean()),
                                       best_validation_jaccard=float(best_validation_jaccard),
                                       best_test_jaccard=float(best_test_jaccard),
                                       best_test_jaccard_full_size=float(best_test_jaccard_full_size),
                                       timestamp=f"{datetime.datetime.now().isoformat()}")

                        logger.info(f"Outputting to weights file {weights_filename} metadata: {json.dumps(json_dict)}")
                        json.dump(json_dict,
                                  checkpoint_info_file)

                if test_jaccards_full_size.mean() > best_test_jaccard_full_size:
                    best_test_jaccard_full_size = test_jaccards_full_size.mean()
                    #######################################################################
                    # Output at the end of each epoch where test accuracy is increasing
                    #######################################################################
                    logger.info('# This step has full size test jaccard better than or equal to the previous best, outputting weights...')
                    os.makedirs(checkpoint_folder, exist_ok=True)
                    weights_filename = 'best_test_full_size'
                    save(model, optimizer, checkpoint_folder, weights_filename)

                    logger.info(f"open(os.path.join({checkpoint_folder}, '{weights_filename}_metadata.json'), 'w'")
                    with open(os.path.join(checkpoint_folder, f'{weights_filename}_metadata.json'), 'w') as checkpoint_info_file:
                        json_dict = dict(epoch=epoch,
                                       exp_id=exp_id,
                                       train_jaccard=float(train_jaccards.mean()),
                                       validation_jaccard=float(validation_jaccards.mean()),
                                       test_jaccard=float(test_jaccards.mean()),
                                       best_validation_jaccard=float(best_validation_jaccard),
                                       best_test_jaccard=float(best_test_jaccard),
                                       best_test_jaccard_full_size=float(best_test_jaccard_full_size),
                                       timestamp=f"{datetime.datetime.now().isoformat()}")

                        logger.info(f"Outputting to weights file {weights_filename} metadata: {json.dumps(json_dict)}")
                        json.dump(json_dict,
                                  checkpoint_info_file)

                #######################################################################
                # Output at the end of each epoch so we can resume training
                #######################################################################
                os.makedirs(checkpoint_folder, exist_ok=True)
                weights_filename = 'checkpoint'
                save(model, optimizer, checkpoint_folder, weights_filename)

                logger.info(f"open(os.path.join({checkpoint_folder}, '{weights_filename}_metadata.json'), 'w'")

                with open(os.path.join(checkpoint_folder, f'{weights_filename}_metadata.json'), 'w') as checkpoint_info_file:
                    json_dict = dict(epoch=epoch,
                            exp_id=exp_id,
                            train_jaccard=float(train_jaccards.mean()),
                            validation_jaccard=float(validation_jaccards.mean()),
                            test_jaccard=float(test_jaccards.mean()),
                            best_validation_jaccard=float(best_validation_jaccard),
                            best_test_jaccard=float(best_test_jaccard),
                            best_test_jaccard_full_size=float(best_test_jaccard_full_size),
                            timestamp=f"{datetime.datetime.now().isoformat()}")

                    logger.info(f"Outputting to weights file {weights_filename} metadata: {json.dumps(json_dict)}")
                    json.dump(json_dict,
                              checkpoint_info_file)

                logger.info('###############################################################')
                logger.info(f'# Finished epoch {epoch}')
                logger.info('###############################################################')
    total_finish = time.time()
    logger.info(f'perf,total,{total_finish - total_start}')
