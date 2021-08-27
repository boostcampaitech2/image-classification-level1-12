import pandas as pd
import torch
import os
df = pd.read_csv('/opt/ml/repos/project_notemplate_LJH/results/densenet_epoch2_lr1e-05_batchsize32_freezeFalse_to_layer_one.csv')

for _ in range(df.shape[0]):
    po = torch.tensor([float(x) for x in df.ans[_].split('[')[1].split(']')[0].split(',')])
    df.ans[_] = torch.argmax(po, dim = -1).item()


print(df.head())
savedir = '/opt/ml/repos/project_notemplate_LJH/results/'
filename ='rm.csv'
df.to_csv(os.path.join(savedir, filename), index = False)