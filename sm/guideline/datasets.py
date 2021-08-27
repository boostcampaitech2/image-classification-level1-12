import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MakeLocateDataFrame():
    
    def __init__(self, type: bool):

        self.type = type
        self.root = r'/opt/ml/data/'
        if self.type:
            self.root += 'train'
        else:
            self.root += 'eval'

    def get_csv(self):
        datas = None

        if self.type:
            datas = pd.read_csv(self.root+'/train.csv')
            datas = datas.drop(['race','id'],axis=1)

        else:
            datas = pd.read_csv(self.root+'/info.csv')

        return datas

    def make_image_locate_frame(self, datas):
        
        labeled_dict = None
        #train = True 인 경우
        if self.type:
            # 이미지 경로를 담은 리스트 생성
            image_locate = [self.root+'/images/'+i for i in datas['path']]
            images = []
            # 경로 안에 있는 숨김 파일은 제거하고 이미지만 저장
            for i in image_locate:
                temp = []
                for j in os.listdir(i):
                    if not j[0] == '.':
                        temp.append(i+'/'+j)
                images.append(temp)
            # 파일 이름으로 클래스를 구분하기 위한 문자
            file_name_string = 'min'
            #새롭게 반환할 데이터 프레임의 틀
            labeled_dict = {"label": [],
                            "gender": [],
                            "age": [],
                            "locate": []}

            for image in images:
                for j in image:
                    #/opt/ml/data/train/images/000002_female_Asian_52/normal.jpg 자른 것
                    get_locate_file = j.split('/')
                    #가장 끝의 파일은 이미지 이름이다.
                    label = file_name_string.index(get_locate_file[-1][0])
                    start = label * 6
                    #000002_female_Asian_52 잘라서 굳이 위에 선언한 데이터 불러오지 않고 수행
                    _, sex, _, age = get_locate_file[-2].split('_')
                    age = int(age)
                    if sex == 'female':
                        start +=3
                    if age < 30:
                        pass
                    elif age < 60:
                        start +=1
                    else:
                        start +=2
                    labeled_dict['label'].append(start)
                    labeled_dict['gender'].append(sex)
                    labeled_dict['age'].append(age)
                    labeled_dict['locate'].append(j)

            labeled_dict = pd.DataFrame(labeled_dict, columns=labeled_dict.keys())

        else:
            datas['locate'] = [self.root+'/images/'+i for i in datas['ImageID']]
            labeled_dict = datas

        return labeled_dict

class MaskDataset(Dataset):
    def __init__(self, data, transform= None, train=True):
        self.data = data
        self.classes = self.data.columns.values.tolist()
        self.transform = transform
        self.train = train

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, idx):
        X = Image.open(self.data['locate'].iloc[idx])

        if self.transform is not None:
            X = self.transform(X)
    
        if self.train:
            y = self.data['label'].iloc[idx]
            return X,y
        
        else:
            return X