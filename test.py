import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_process.dataset as module_dataset
import data_process.transform as module_transform
import model as module_model
import loss as module_loss
import metric as module_metric
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_process instances
    dataset = config.init_obj('dataset', module_dataset)
    dataset.transform = getattr(module_transform, config['transform'])
    dataset.target_transform = getattr(module_transform, config['target_transform'])
    data_loader = DataLoader(dataset, **config['data_loader'])

    # build model architecture
    model = config.init_obj('model', module_model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

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

    total_metrics = torch.zeros(len(metric_fns))
    batch_outputs = list()
    batch_targets = list()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_outputs.append(output.detach())
            batch_targets.append(target.detach())
        total_output = torch.cat(batch_outputs)
        total_target = torch.cat(batch_targets)
        # computing loss, metrics on test set
        total_loss = loss_fn(total_output, total_target).item()
        for i, metric in enumerate(metric_fns):
            total_metrics[i] += metric(total_output, total_target)

    n_samples = len(dataset)
    log = {'loss': total_loss / n_samples}
    log.update({
        # met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./config_test.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
