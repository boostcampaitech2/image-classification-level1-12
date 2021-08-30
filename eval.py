import argparse
import os
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_process.dataset as module_dataset
import data_process.transform as module_transform
import model as module_model
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('eval')

    # setup data_process instances
    dataset = config.init_obj('dataset', module_dataset)
    dataset.transform = getattr(module_transform, config['transform'])
    data_loader = DataLoader(dataset, shuffle=False, **config['data_loader'])

    # build model architecture
    model = config.init_obj('model', module_model)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config['checkpoint_path']))
    checkpoint = torch.load(config['checkpoint_path'])
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions: List[int] = list()
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data)
            pred = torch.argmax(output, dim=1)
            all_predictions.extend(pred.detach().cpu().tolist())

    submission = pd.read_csv(config["dataset"]["args"]["csv_path"])
    submission['ans'] = all_predictions

    os.makedirs(os.path.dirname(config["submisson_path"]), exist_ok=True)
    submission.to_csv(config["submisson_path"], index=False)

    logger.info(f'{os.path.abspath(config["submisson_path"])} is saved.')
    logger.info("Evaluation is done!")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./config_eval.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
