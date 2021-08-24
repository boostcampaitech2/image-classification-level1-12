import argparse
import collections
import torch
import numpy as np
from data_loader.data_loaders import MaskDataLoader, MaskDataset, collate_fn, sub_collate_fn
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, CV, dirlister, to_label

import pandas as pd
import os


# fix random seeds for reproducibility
SEED = 49
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
    SUB_DATA_ROOT = '/opt/ml/input/data/eval/'
    train_meta = pd.read_csv(os.path.join(TRAIN_DATA_ROOT, 'train.csv'))
    sub_meta = pd.read_csv(os.path.join(SUB_DATA_ROOT, 'info.csv'))

    train_dir_list = dirlister(TRAIN_DATA_ROOT, train_meta)
    train_cv = CV(train_dir_list, 5)

    sub_dir_list = dirlister(SUB_DATA_ROOT, sub_meta, mode = 'test')
    sub_dataloader = MaskDataLoader(MaskDataset(sub_dir_list, meta = sub_meta), 1, collate_fn=sub_collate_fn)

    for idx, (train, valid, test) in enumerate(train_cv):

        print(f'start fold {idx + 1}/{train_cv.maxfold}')

        train_dataloader = MaskDataLoader(MaskDataset(train, meta = train_meta), 1, collate_fn=collate_fn)
        valid_dataloader = MaskDataLoader(MaskDataset(valid, meta = train_meta), 1, collate_fn=collate_fn)
        test_dataloader = MaskDataLoader(MaskDataset(test, meta = train_meta), 1, collate_fn=collate_fn)

        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        train_data_loader=train_dataloader,
                        valid_data_loader=valid_dataloader,
                        test_data_loader = test_dataloader,
                        lr_scheduler=lr_scheduler)

        trainer.train()

        #test epoch 구현 필요
    #submission collate fn 구현 필요
    #utils.to_label 구현 필요
    out = []
    for batch in sub_dataloader:
        mask, age, gender = model(batch)
        out.append(to_label(mask, age, gender))

    out_csv = sub_meta.copy()
    out_csv['ans'] = out
    out_csv.to_csv('/opt/ml/repos/project_LJH/submission.py', index = False)
    print('complete')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='/opt/ml/repos/project_LJH/config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='/opt/ml/repos/project_LJH/ckpt', type=str,
                      help='path to latest checkpoint')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
