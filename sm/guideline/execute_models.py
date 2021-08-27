import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.optim as optim
from PIL import Image
import argparse
from datasets import MaskDataset, MakeLocateDataFrame
from models import MaskModel
from loss import Loss
import torch
import tqdm
import time
import os

def main():
    NUM_CLASSES = 18
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description="command line options")

    parser.add_argument('--mode', type=str, default='train', help='set mode to execute models (default : train)')
    parser.add_argument('--lr', type=float, default=0.001, help='set learning rate (default : 0.001)')
    parser.add_argument('--seed', type=int, default=12, help='set random seed number (default : 12)')

    args = parser.parse_args()
    
    if args.mode == 'train':
        low_datas = MakeLocateDataFrame(True)
        data_frames = low_datas.make_image_locate_frame(low_datas.get_csv())
        transform = transforms.Compose([
                    # 선형 보간법을 사용하여 출력 픽셀값을 계산 -> 부드러운 느낌을 준다는데 누구 기준인진 잘
                    transforms.Resize((512,384), Image.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                    ])
        datasets = MaskDataset(data_frames, transform=transform,train=True)
        data_loader = DataLoader(datasets, batch_size=64, shuffle=True, num_workers=2)
        model = MaskModel(18).to(DEVICE)
        model.init_params()
        criterion = Loss('default').loss_function()
        optm = optim.Adam(model.parameters(), lr=args.lr)

        start_time = time.time()
        for epoch in range(5):
            loss_value_sum = 0
            epoch_f1 = 0
            # 바로 data_loader 값을 넣을 수 있음
            for i, (images, labels) in enumerate(tqdm.tqdm(data_loader, leave=False)):
                labels = torch.tensor(list(labels)).to(DEVICE)
                # gradient 초기화 : 이전 epoch 의 결과가 반영되면 안되기 때문 
                optm.zero_grad()
                logits = model.forward(images.to(DEVICE))
                loss_out = criterion(logits, labels)

                # get f1_score
                _,predict = torch.max(logits,1)
                epoch_f1 += f1_score(labels.cpu().numpy(), predict.cpu().numpy(), average='macro')

                if i % 80 == 0:
                    print(f'\tepoch : {epoch} loss {i} : {loss_out.data}')
                loss_out.backward()
                optm.step()
                loss_value_sum += loss_out

            print(f'loss avg : {loss_value_sum/len(data_loader)}')
            print(f'f1_score per {epoch} : {epoch_f1/epoch:4f}')
        end_time = time.time()
        print(f'{(end_time - start_time)//60} minutes')

        if not os.path.exists('./model_results'):
            os.mkdir('./model_results')

        now = time.strftime('%m_%d_%H_%M',time.localtime(end_time))
        torch.save(model.state_dict(),'./model_results/'+f'model_{now}.pt')

    else:
        print('hello')

if __name__ == '__main__':
    main()