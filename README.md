# ๐ท Boostcamp AI Tech P-Stage 1 - Image Classification

2021.08.23 ~ 2021.09.03 [Wrap-up Report](https://bit.ly/38EbbIw)

## ๐ Summary
### Mask status Classification

>    COVID-19์ ํ์ฐ์ผ๋ก ์ฐ๋ฆฌ๋๋ผ๋ ๋ฌผ๋ก  ์  ์ธ๊ณ ์ฌ๋๋ค์ ๊ฒฝ์ ์ , ์์ฐ์ ์ธ ํ๋์ ๋ง์ ์ ์ฝ์ ๊ฐ์ง๊ฒ ๋์์ต๋๋ค. ์ฐ๋ฆฌ๋๋ผ๋ COVID-19 ํ์ฐ ๋ฐฉ์ง๋ฅผ ์ํด ์ฌํ์  ๊ฑฐ๋ฆฌ ๋๊ธฐ๋ฅผ ๋จ๊ณ์ ์ผ๋ก ์ํํ๋ ๋ฑ์ ๋ง์ ๋ธ๋ ฅ์ ํ๊ณ  ์์ต๋๋ค. ๊ณผ๊ฑฐ ๋์ ์ฌ๋ง๋ฅ ์ ๊ฐ์ง ์ฌ์ค(SARS)๋ ์๋ณผ๋ผ(Ebola)์๋ ๋ฌ๋ฆฌ COVID-19์ ์น์ฌ์จ์ ์คํ๋ ค ๋น๊ต์  ๋ฎ์ ํธ์ ์ํฉ๋๋ค. ๊ทธ๋ผ์๋ ๋ถ๊ตฌํ๊ณ , ์ด๋ ๊ฒ ์ค๋ ๊ธฐ๊ฐ ๋์ ์ฐ๋ฆฌ๋ฅผ ๊ดด๋กญํ๊ณ  ์๋ ๊ทผ๋ณธ์ ์ธ ์ด์ ๋ ๋ฐ๋ก COVID-19์ ๊ฐ๋ ฅํ ์ ์ผ๋ ฅ ๋๋ฌธ์๋๋ค.  
>    ๊ฐ์ผ์์ ์, ํธํก๊ธฐ๋ก๋ถํฐ ๋์ค๋ ๋น๋ง, ์นจ ๋ฑ์ผ๋ก ์ธํด ๋ค๋ฅธ ์ฌ๋์๊ฒ ์ฝ๊ฒ ์ ํ๊ฐ ๋  ์ ์๊ธฐ ๋๋ฌธ์ ๊ฐ์ผ ํ์ฐ ๋ฐฉ์ง๋ฅผ ์ํด ๋ฌด์๋ณด๋ค ์ค์ํ ๊ฒ์ ๋ชจ๋  ์ฌ๋์ด ๋ง์คํฌ๋ก ์ฝ์ ์์ ๊ฐ๋ ค์ ํน์ ๋ชจ๋ฅผ ๊ฐ์ผ์๋ก๋ถํฐ์ ์ ํ ๊ฒฝ๋ก๋ฅผ ์์ฒ ์ฐจ๋จํ๋ ๊ฒ์๋๋ค. ์ด๋ฅผ ์ํด ๊ณต๊ณต ์ฅ์์ ์๋ ์ฌ๋๋ค์ ๋ฐ๋์ ๋ง์คํฌ๋ฅผ ์ฐฉ์ฉํด์ผ ํ  ํ์๊ฐ ์์ผ๋ฉฐ, ๋ฌด์ ๋ณด๋ค๋ ์ฝ์ ์์ ์์ ํ ๊ฐ๋ฆด ์ ์๋๋ก ์ฌ๋ฐ๋ฅด๊ฒ ์ฐฉ์ฉํ๋ ๊ฒ์ด ์ค์ํฉ๋๋ค. ํ์ง๋ง ๋์ ๊ณต๊ณต์ฅ์์์ ๋ชจ๋  ์ฌ๋๋ค์ ์ฌ๋ฐ๋ฅธ ๋ง์คํฌ ์ฐฉ์ฉ ์ํ๋ฅผ ๊ฒ์ฌํ๊ธฐ ์ํด์๋ ์ถ๊ฐ์ ์ธ ์ธ์ ์์์ด ํ์ํ  ๊ฒ์๋๋ค.  
>    ๋ฐ๋ผ์, ์ฐ๋ฆฌ๋ ์นด๋ฉ๋ผ๋ก ๋น์ถฐ์ง ์ฌ๋ ์ผ๊ตด ์ด๋ฏธ์ง ๋ง์ผ๋ก ์ด ์ฌ๋์ด ๋ง์คํฌ๋ฅผ ์ฐ๊ณ  ์๋์ง, ์ฐ์ง ์์๋์ง, ์ ํํ ์ด ๊ฒ์ด ๋ง๋์ง ์๋์ผ๋ก ๊ฐ๋ ค๋ผ ์ ์๋ ์์คํ์ด ํ์ํฉ๋๋ค. ์ด ์์คํ์ด ๊ณต๊ณต์ฅ์ ์๊ตฌ์ ๊ฐ์ถฐ์ ธ ์๋ค๋ฉด ์ ์ ์ธ์ ์์์ผ๋ก๋ ์ถฉ๋ถํ ๊ฒ์ฌ๊ฐ ๊ฐ๋ฅํ  ๊ฒ์๋๋ค.

### Labeling
* ๋ง์คํฌ ์ฐฉ์ฉ ์ฌ๋ถ : ์ฐฉ์ฉ / ์๋ชป๋ ์ฐฉ์ฉ(ํฑ์คํฌ or ์ฝ์คํฌ) / ๋ฏธ์ฐฉ์ฉ
* ์ฑ๋ณ : ๋จ / ์ฌ
* ์ฐ๋ น : 30๋ ๋ฏธ๋ง / 30๋ ์ด์ ~ 60๋ ๋ฏธ๋ง / 60๋ ์ด์

์ด 18๊ฐ์ label ๋ถ๋ฅ  

<img src='https://user-images.githubusercontent.com/54921730/132083902-edabcce7-4ff7-4517-a1a5-f840fa86b57b.png' width=900 height=700/>

## ๐จโ๐ฌResult
<img src='https://user-images.githubusercontent.com/54921730/132083921-9799aecc-9185-4779-943b-289e4c4f757e.png' alt='132083921-9799aecc-9185-4779-943b-289e4c4f757e' width=900/>  

* ์ฃผ์ด์ง ๋ฐ์ดํฐ๋ ๊ณต๊ฐ๊ฐ ๋ถ๊ฐํ์ฌ ๋ด ์ฌ์ง๊ณผ ํ์ ์ฌ์ง, ์ ์๊ถ ๊ณต๊ฐ๋ ์ฌ์ง์ ์ด์ฉํด์ ์คํ
* ํ์ต ๋ฐ์ดํฐ์ ๋น์ทํ ๊ฒฝ์ฐ ์ญ์๋ ์ ๋ง์ถค
  * ํ์ง๋ง ์ค์  ๊ฒฐ๊ณผ์ ๋น์ทํ๊ฒ ๋ง์คํฌ๋ก ๋๊ณผ ์ฝ๋ฅผ ๊ฐ๋ฆฐ ๊ฒฝ์ฐ ๋ง์ถ๊ธฐ ํ๋ค์ด ํจ 
* ํน์ดํ ์ ์ ํ์ต ๋ฐ์ดํฐ์ ์ ํ ์๋ ์์ฃผ ์ด๋ฆฐ ์์ด๋ ๋ง์ท๋ค๋ ์ฌ์ค๊ณผ, ์์ผ๊ธฐ๋ฅธ ๋ฐ์ดํฐ๊ฐ ์๋ค๋ณด๋ ๋ง์คํฌ๋ก ์ธ์ํ ์ ์ด๋ค.

* License - https://pixabay.com/, MyImage, ์บ ํผ ์กฐํ๋

## ๐ Introduction Team

---

<img src="https://raw.githubusercontent.com/herjh0405/Img/master/img/KakaoTalk_20210902_105117185.png" alt="KakaoTalk_20210902_105117185"/>

|                            ํ์ ํ                            |                            ์์ฑ๋ฏผ                            |                            ์กฐํ๋                            |                            ํฉ์์                            |                            ์ค์ฃผ์                            |                            ์ด์คํ                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![Avatar](https://avatars.githubusercontent.com/u/54921730?v=4)](https://github.com/herjh0405) | [![Avatar](https://avatars.githubusercontent.com/u/49228132?v=4)](https://github.com/mickeyshoes) | [![Avatar](https://avatars.githubusercontent.com/u/61579014?v=4)](https://github.com/JODONG2) | [![Avatar](https://avatars.githubusercontent.com/u/49892621?v=4)](https://github.com/WonsangHwang) | [![Avatar](https://avatars.githubusercontent.com/u/69762559?v=4)](https://github.com/Jy0923) | [![Avatar](https://avatars.githubusercontent.com/u/49234207?v=4)](https://github.com/kmouleejunhyuk) |
|        [์ฃผ์๊ต ์ ์](https://herjh0405.tistory.com/)         |   [์ค์ฃผ์<br/>๊ทธ๋ ์ ์ด์ผ](https://velog.io/@mickeyshoes)    | [zi์กด๋ถ์บ ](https://shimmering-form-67a.notion.site/WEEK-e0a8cfccd85a43fca143a14641de8e30) |                         ์ดํ์ ๋ฉํ                           |                           ์ ํ์ฐกโค                            | [์ต๊ณ ์<br/>๋ชจ๋ธ๋ฌ<br/>ํ์ ํ](https://dreaming-lee.notion.site/boostcamp-archive-44d6ea71b8bf4c0e9dc8d37e57ebbf5f) |
|              `๋ฐ์ดํฐ๋ถ์` `CV`<br>  `์์ฑ์ธ์`               |                `CV` `๋ชจ๋ธ ์๋น` <br> `๋ฐฑ์๋`                |                        `CV` `AutoML`                         |                             `CV`                             |                   `CV` `Generative Model`                    |                             `CV`                             |

## ๐Data Preprocessing
### Face Crop

* `cvlib` - By cvlib.detect_face, crop face+cloth coordinate(x, y, w, h)

## ๐จ๐พโ๐ปModel
### Backbone

* `ResNet18`(https://pytorch.org/hub/pytorch_vision_resnet/) - model fine-tuning

### Loss

* `Focal loss` (gamma = 2)

### Optimizer

* `Adam`

### Albumentation

* `Resize `
* `GaussianBlur`
* `Normalize`
* `HorizontalFlip`

### Wandb log
![image-20210902192217807](https://raw.githubusercontent.com/herjh0405/Img/master/img/image-20210902192217807.png)

## ๐ฏPerformance
### Public

* F1 Score : 0.763
* Accuracy : 79.206%

### Private

* F1 Score : 0.744
* Accuracy : 78.333%

## ๐ปHardware

* Ubuntu 18.04.5 LTS
* Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz ์ค ๋ผ๋ฆฌ์ฝ์ด 8๊ฐ
* Ram 90GB
* Tesla V100 32GB

## ๐ฎGetting Started

### File Structure

```text
pro_hun/
โโโ base
โ  โโโ baseline.ipynb		# Introduction & EDA Baseline
โโโ data_preprocessing
โ  โโโ data_labeling.py		# Data labeling(18 classes&Error Fix)
โ  โโโ data_split.py		# Apply StratifiedKFold
โ  โโโ image_crop.py		# Crop face by cvlib
โโโ experiment
โ  โโโ model_test.ipynb		# check submission distribution
โโโ model
โ  โโโ loss.py			# Focal loss
โ  โโโ metric.py		# Acc, F1_score
โ  โโโ model.py
โโโ requirements.txt
โโโ test.py
โโโ train.py
โโโ utils
  โโโ util.py			# For smooth processing
```

### Dependencies

* torch==1.7.1
* torchvision==0.8.2
* cvlib==0.2.6
* opencv-python==4.5.3.56
* albumentations==1.0.3
* matplotlib==3.2.1
* numpy==1.19.5
* Pillow==8.3.1
* pandas==1.1.5
* scikit-learn==0.24.2
* tqdm==4.62.2
* wandb==0.12.1

### Install Requirements
```
$ pip install -r requirements.txt
```
### Data Preprocessing

* labeling, crop, split, 

```
$ python data_preprocessing/data_labeling.py
$ python data_preprocessing/image_crop.py
$ python data_preprocessing/data_split.py
```

### Training

```
$ python train.py [-lr] [-bs] [--epoch] [--train_path] [--model_save] [--image_data] [--image_dir]
```

* `-lr` : learning rate, default=0.0001
* `-bs` : batch size, default=128
* `--epoch` : Epoch size, default=10
* `--train_path` : train_directory_path
* `--model_save` : model_save_path
* `--image_data` : CSV according to image type(Original, Crop, All), default='train_with_all.csv'
* `--image_dir` : Directory according to image type, default='ori_image_all'

### Evaluation
```
$ python test.py [--test_path] [--result_save] [--image_dir]
```

* `--test_path` : test_directory_path
* `--result_save` : submission_save_path
* `--image_dir` : Directory according to image type, default='crop_image'

## Acknowledgements
๋ณธ ํ๋ก์ ํธ๋ Victor Huang ์ PyTorch Template Project ์ ๊ธฐ๋ฐ์ผ๋ก ํ์ฌ ๊ฐ๋ฐ๋์์ต๋๋ค.

## License
This project is licensed under the MIT License. See LICENSE for more details.



















