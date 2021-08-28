import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader.data_sets import MaskDataset
from model.model import MaskModel
import model.metric as module_metric
from model.loss import MaskLoss
from utils import prepare_device
from parse_config import ConfigParser


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)


def convert_3class_to_1class(output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mask_pred = torch.argmax(output[0], dim=1)
    gender_pred = torch.argmax(output[1], dim=1)
    age_pred = torch.argmax(output[2], dim=1)

    return torch.mul(mask_pred, 6) + torch.mul(gender_pred, 3) + age_pred


def main(config):
    logger = config.get_logger('eval')

    # define image transform
    eval_transform = transforms.Compose([
        # transforms.Scale(244),
        transforms.CenterCrop((384, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    eval_dataset = MaskDataset("/opt/ml/mask_data",
                                train=False,
                                transform=eval_transform
                                )

    eval_data_loader = DataLoader(eval_dataset,
                                  batch_size=64,
                                  shuffle=False,
                                  num_workers=4)

    # build model architecture
    model = MaskModel()
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = MaskLoss()
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    checkpoint_file_path = "/tmp/pycharm_project_862/saved/models/Mask_vgg19_lin2/0828_133524/checkpoint-epoch8.pth"
    logger.info('Loading checkpoint: {} ...'.format(checkpoint_file_path))
    checkpoint = torch.load(checkpoint_file_path)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions = list()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(eval_data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            all_predictions.extend(convert_3class_to_1class(output).cpu().tolist())

    # 제출할 파일을 저장합니다.
    submission = pd.read_csv("/opt/ml/mask_data/eval/info.csv")
    submission['ans'] = all_predictions
    submission.to_csv('./saved/submission/submission_0828_133524_8.csv', index=False)
    # epoch: 3
    # loss: 0.2146139108273299
    # mask_total_accuracy: 0.9237868785858154
    # mask_accuracy: 0.9955168776371308
    # gender_accuracy: 0.9864847046413502
    # age_accuracy: 0.9404008438818565
    # val_loss: 0.9550173132369916
    # val_mask_total_accuracy: 0.8796875476837158
    # val_mask_accuracy: 0.98984375
    # val_gender_accuracy: 0.9752604166666666
    # val_age_accuracy: 0.9096354166666667
    print('test inference is done!')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='./config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
