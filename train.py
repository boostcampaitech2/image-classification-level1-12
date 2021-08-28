import argparse
import collections
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from data_loader.data_sets import MaskDataset
from model.model import MaskModel
from model.loss import MaskLoss
# from model.loss import mask_total_loss


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('train')

    # define image transform
    train_transform = transforms.Compose([
        # transforms.Scale(244),
        transforms.CenterCrop(244),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    valid_transform = transforms.Compose([
        # transforms.Scale(244),
        transforms.CenterCrop(244),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # setup data_loader instances
    # data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation()
    train_dataset = MaskDataset("/opt/ml/mask_data",
                                train=True,
                                num_folds=5,
                                folds=[0, 1, 2, 3],
                                transform=train_transform
                                )
    valid_dataset = MaskDataset("/opt/ml/mask_data",
                                train=True,
                                num_folds=5,
                                folds=[4],
                                transform=valid_transform
                                )
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=8)
    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=64,
                                   shuffle=True,
                                   num_workers=8)



    # build model architecture, then print to console
    # model = config.init_obj('arch', module_arch)
    # logger.info(model)
    model = MaskModel()
    # print(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # device = 'cpu'

    # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    # criterion = mask_total_loss
    mask_weight = None
    gender_weight = None
    # age_weight = torch.tensor([1.4, 1., 6.1]).to(device)
    # age_weight = torch.tensor([1.2, 1., 3.4]).to(device)
    age_weight = torch.tensor([1.2, 1., 6]).to(device)
    criterion = MaskLoss(
                    mask_weight=mask_weight,
                    gender_weight=gender_weight,
                    age_weight=age_weight
                )

    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
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
