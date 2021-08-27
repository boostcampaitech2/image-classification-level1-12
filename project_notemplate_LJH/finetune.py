import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/opt/ml/repos/project_notemplete_LJH/')
import loss


def tuner(epoch, lr, batch_size, device, dataloader, model):
    # fix random seeds for reproducibility
    SEED = 49
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    criterion = loss.F1_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('start finetune')
    torch.cuda.empty_cache()
    for e in range(epoch):
        model.train()

        for param in model.backbone.features.parameters():
            param.requires_grad = False

        for idx, (img, age, gender, mask) in tqdm(enumerate(dataloader), total = len(dataloader)//batch_size + 1, leave=False, desc = f'    epoch {e}: '):
            optimizer.zero_grad()
            img, age, gender, mask = img.to(device), age.to(device),gender.to(device),mask.to(device)

            age_pred, gender_pred, mask_pred  = model(img)
            loss_age, loss_mask, loss_gender = criterion(age_pred, age), criterion(mask_pred, mask), criterion(gender_pred, gender)

            loss_sum = loss_age+loss_mask+loss_gender
            loss_sum.backward()
            optimizer.step()


    print('finetune complete')
    del img, age, gender, mask
    torch.cuda.empty_cache()    

    return model.state_dict()