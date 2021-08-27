import os
import pandas as pd
import tqdm


class Make_Label:
    def __init__(self, BASE_PATH: str, CSV_FILE: pd.DataFrame, COLS: list):
        """
        기존의 train.csv 파일에 labeling을 새로 해주기 위해 경로 설정
        Args:
            BASE_PATH (str): train.csv File path
            CSV_FILE (pd.DataFrame): train.csv
            COLS (list): 새로 만들 DataFrame의 COLUMN
        """
        self.path = BASE_PATH
        self.img_path = os.path.join(BASE_PATH, "images")
        self.csv_file = CSV_FILE
        self.df = pd.DataFrame(columns=COLS)

    def labeling(self):
        idx_df = 0
        # 이미지 폴더 path
        for idx in tqdm.tqdm(range(train_df.shape[0])):
            IMG_PATH = os.path.join(self.img_path, train_df.iloc[idx]["path"])
            file_list = os.listdir(IMG_PATH)

            for file in file_list:
                if file.rstrip().startswith("._"):  # ._{파일명} 제거
                    continue

                file_path = os.path.join(IMG_PATH, file)
                # train.csv 파일에서 gender, age 정보를 가져옴
                self.df.loc[idx_df] = train_df.loc[idx][["gender", "age"]]
                # image 파일 전체 경로, 이름
                self.df.loc[idx_df][["path", "name"]] = [
                    file_path,
                    file_path.split("/")[-2] + "_" + file,
                ]
                self.check_label(self.df, idx_df)
                idx_df += 1

        # 새로 만든 DataFrame 저장
        labeling_data_path = "/opt/ml/image-classification-level1-12/templates/data/train/train_with_label.csv"
        self.df.to_csv(labeling_data_path, index=False)

    # 대회 기준에 맞춰서 label 부여
    # 여자일 경우+3, 마스크 착용 상태가 incorrect+6, Normal+12
    def check_label(self, df, idx_df):
        mask = df.loc[idx_df]["name"]
        gender = df.loc[idx_df]["gender"]
        age = df.loc[idx_df]["age"]

        if age < 30:
            label = 0
        elif 30 <= age < 60:
            label = 1
        else:
            label = 2
        if gender == "female":
            label += 3
        if "incorrect" in mask:
            label += 6
        elif "normal" in mask:
            label += 12

        df.loc[idx_df]["label"] = label


if __name__ == "__main__":
    # train 데이터 경로 지정
    train_path = "/opt/ml/image-classification-level1-12/templates/data/train"
    # train 데이터를 데이터프레임 형태로
    train_df = pd.read_csv(os.path.join(train_path, "train.csv"))

    # labeling 시행
    make_label = Make_Label(
        train_path, train_df, ["gender", "age", "path", "name", "label"]
    )
    make_label.labeling()

    # 에러 데이터 처리 코드 -> 결과가 오히려 안좋아져서 보류
    # df = pd.read_csv('/opt/ml/image-classification-level1-12/templates/data/train/train_with_label.csv')

    # df['error_check'] = df['name'].map(lambda x : 1
    #             if ('006359' in x or '006360' in x or '006361' in x or '006362' in x or '006363' in x or '006364' in x)
    #             else (2 if ('001498-1' in x or '004432' in x)
    #             else (3 if ('000020' in x or '004418' in x or '005227' in x) and ('incorrect'  in x or 'normal'in x) else 0)))

    # df['label'] = df.apply(lambda x: x['label']-3 if x['error_check']==1
    #                             else (x['label']+3 if x['error_check']==2
    #                             else (x['label']+6 if (x['error_check']==3 and 'incorrect' in x['name'])
    #                             else (x['label']-6 if (x['error_check']==3 and 'normal' in x['name']) else x['label']))), axis=1)

    # df.drop(columns=['error_check'], inplace=True)
    # df.to_csv("/opt/ml/image-classification-level1-12/templates/data/train/train_with_label.csv", index=False)
