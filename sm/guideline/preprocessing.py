import os
import pandas as pd
import albumentations as A
import cv2

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

    def over_60_data_augmentation(self, data):
        #get datas over 60
        over_60 = data.query('age==60')

        if not over_60.empty:
            ORIGIN_FILENAME = ("normal", "mask1", "mask2", "mask3", "mask4", "mask5", "incorrect_mask")
            REWRITE_FILENAME = ("normal_hf", "mask1_hf", "mask2_hf", "mask3_hf", "mask4_hf", "mask5_hf", "incorrect_mask_hf")
            file_rename = []

            transform = A.Compose([
                # A.RandomCrop(width=256, height=256),
                A.HorizontalFlip(p=1),
                A.Cutout(num_holes=1,max_h_size=100, max_w_size=100,p=0.7)
            ])

            # transform image to save
            for image_locate in over_60['locate']:
                img = cv2.imread(image_locate)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_transform = transform(image=img)
                
                # make locate to save image
                loc_split = image_locate.split('/')
                filename, extension = loc_split[-1].split('.')
                loc_split[-1] = REWRITE_FILENAME[ORIGIN_FILENAME.index(filename)]+'.'+extension
                rename_locate = '/'.join(loc_split)
                file_rename.append(rename_locate)
                # albumentation 처리된 후에 dict 로 반환되어 이미지만 따로 빼야함
                cv2.imwrite(filename=rename_locate, img=image_transform['image'])
            
            over_60['locate'] = file_rename
            total_data = data.append(over_60)
            return total_data

        else:
            raise ValueError("You should put labeled csv file.")


