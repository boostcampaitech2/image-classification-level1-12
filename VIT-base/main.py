from torch import torch
import datetime
import numpy as np
from train import sub_session, train_session


if __name__ == '__main__':
    # fix random seeds for reproducibility
    SEED = 20
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    #device = torch.device('cuda:0' if torch.cuda.device_count() > 0 else 'cpu')

    TRAIN_DATA_ROOT = '/opt/ml/input/data/train/'
    SUB_DATA_ROOT = '/opt/ml/input/data/eval/'
    epoch = 10
    batch_size = 20
    lr = 1e-5
    cv_num = 3
    freeze = False
    debug = False
    addition = 'to_layer_one'

    print(f'\nDEBUG: {debug}')
    print(f'estimated end time: {datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))) + datetime.timedelta(minutes=10*(epoch*cv_num + 1) + 3)}')

    model = train_session(epoch, batch_size, lr, debug)
    torch.save(model.state_dict(), '/opt/ml/repos/VIT-base/result/model.pt')

    result = sub_session(model, epoch, batch_size, debug)
    result.to_csv('/opt/ml/repos/VIT-base/result/submission.csv')

    print('done')
    