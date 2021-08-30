import argparse
import collections

import torch
from torch.utils.data import DataLoader
import numpy as np

import data_process.dataset as module_dataset
import data_process.transform as module_transform
import model as module_model
import loss as module_loss
import metric as module_metric

from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # setup data_process instances
    training_dataset = config.init_obj('training_dataset', module_dataset)
    training_dataset.transform = getattr(module_transform, config['training_transform'])
    training_dataset.target_transform = getattr(module_transform, config['train_target_transform'])
    training_data_loader = DataLoader(training_dataset, **config['training_data_loader'])

    valid_dataset = config.init_obj('valid_dataset', module_dataset)
    valid_dataset.transform = getattr(module_transform, config['valid_transform'])
    valid_dataset.target_transform = getattr(module_transform, config['train_target_transform'])
    valid_data_loader = DataLoader(valid_dataset, **config['valid_data_loader'])

    # build model architecture, then print to console
    model = config.init_obj('train_model', module_model)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=training_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="./config_train.json", type=str,
                      help='config file path (default: ./config_train.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
