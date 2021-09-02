import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.optim as optim
from PIL import Image
import argparse
import pandas as pd
from datasets import MaskDataset
from preprocessing import MakeLocateDataFrame
from models import MaskModel
from loss import Loss
import torch
import tqdm
import time
import os

def main():
    NUM_CLASSES = 18
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # for filename
    now = time.strftime('%m_%d_%H_%M',time.localtime(time.time()))
    
    # parse argument
    parser = argparse.ArgumentParser(description="command line options")
    parser.add_argument('--mode', type=str, default='train', help='set mode to execute models (default : train)')
    parser.add_argument('--lr', type=float, default=0.0001, help='set learning rate (default : 0.0001)')
    parser.add_argument('--seed', type=int, default=12, help='set random seed number (default : 12)')
    parser.add_argument('--epochs', type=int, default=5, help='set epochs (default :5 , max : 10)')

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
        data_loader = DataLoader(datasets, batch_size=128, shuffle=True)#, num_workers=2)
        model = MaskModel(NUM_CLASSES).to(DEVICE)
        # model.init_params()
        criterion = Loss('default').loss_function()
        optm = optim.Adam(model.parameters(), lr=args.lr)

        # create lr scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optm,
        mode='max',factor=0.5, patience=2)

        if args.epochs > 11:
            args.epochs = 10

        model.train()

        start_time = time.time()
        for epoch in range(args.epochs):
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
                lr_scheduler.step(loss_out/len(data_loader))
                loss_value_sum += loss_out

            print(f'loss avg : {loss_value_sum/len(data_loader)}')
            print(f'f1_score per {epoch} : {epoch_f1/(epoch+1):4f}')
            print(optm.state_dict()['param_groups'][0]['lr'])

            # 이 조건을 valid 로 정해야 하는거 같은데?
            if epoch_f1/(epoch+1) > 90 and loss_value_sum/len(data_loader) <0.7:
                torch.save(model.state_dict(),'./pre_model_results/'+now + f'_{epoch}_stopped.pt')

        end_time = time.time()
        print(f'{(end_time - start_time)//60} minutes')

        if not os.path.exists('./model_results'):
            os.mkdir('./model_results')

        torch.save(model.state_dict(),'./model_results/'+f'model_{now}.pt')

    else:
        # init data
        low_datas = MakeLocateDataFrame(False)
        data_frames = low_datas.make_image_locate_frame(low_datas.get_csv())
        transform = transforms.Compose([
            # 선형 보간법을 사용하여 출력 픽셀값을 계산 -> 부드러운 느낌을 준다는데 누구 기준인진 잘
            transforms.Resize((512,384), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            ])
        datasets = MaskDataset(data_frames, transform = transform, train=False)
        data_loader = DataLoader(datasets, batch_size=128, shuffle=False)
        
        # get submission
        submission = pd.read_csv('./results/info.csv')

        # get model
        model = MaskModel(NUM_CLASSES).to(DEVICE)
        model.load_state_dict(torch.load('./model_results/model_best.pth'))
        
        #test
        with torch.no_grad():
            # 추론(inference)을 하기 전에 model.eval() 메소드를 호출하여 드롭아웃(dropout)과 배치 정규화(batch normalization)를 
            # 평가 모드(evaluation mode)로 설정해야 합니다. 그렇지 않으면 일관성 없는 추론 결과가 생성됩니다.
            model.eval()
            results = []
            for i,image in enumerate(tqdm.tqdm(data_loader, leave=False)):
                logits = model.forward(image.to(DEVICE))

                for predict in logits:
                    results.append(torch.argmax(predict, dim=-1).item())
            
        submission['ans'] = results
        submission.to_csv(f'./results/{now}.csv', index=False)

if __name__ == '__main__':
    main()