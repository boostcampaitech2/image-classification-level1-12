import argparse
import cv2
import shutil
import cvlib as cv
import pandas as pd
import os
import sys


sys.path.append('/opt/ml/image-classification-level1-12')
from utils.util import copy_folder

def crop_coord(imagePath):
	image = cv2.imread(imagePath)
	face, confidence = cv.detect_face(image)
	if len(face)==0 :
		return 0
	x, y, w, h = face[0]
	coord = [x, y, w, h]

	return coord


def crop_face(coord_list, path, crop_path):
	if type(coord_list)==int:
		shutil.copy(path, crop_path)
		return
	x, y, w, h = coord_list
	image = cv2.imread(path)
	H, W, C = image.shape
	image_array = image[max(y-100,0):min(h+100,H), max(0,x-100):min(w+100,W)]
	cv2.imwrite(crop_path, image_array)


if __name__ == '__main__':
	args = argparse.ArgumentParser(description='PyTorch Template')
	args.add_argument("--data_path", default="/opt/ml/image-classification-level1-12/data", type=str, help="data_directory_path",)
	args.add_argument("--data_type", default="train", type=str, help="data type-train or eval")
	args.add_argument("--image_data", default="train_with_label.csv", type=str, help="CSV according to image type(Original, Crop, All)",)
	args.add_argument("--crop_dir", default="crop_images", type=str, help="Crop image dir name",)
	args = args.parse_args()


	dir_path = os.path.join(args.data_path, args.data_type)
	# 기존의 images 폴더 복사
	copy_folder(dir_path, args.crop_dir)
	
	label_path = os.path.join(dir_path, args.image_data)
	df = pd.read_csv(label_path)
	# eval image를 crop
	if 'info' in args.image_data :
		df['crop_coord'] = df.apply(lambda x: crop_coord(dir_path+'/images/'+x['ImageID']),axis=1)
		df.apply(lambda x : crop_face(x['crop_coord'], dir_path+'/images/'+x['ImageID'], dir_path+'/crop_images/'+x['ImageID']), axis=1)
	# train image를 crop
	else:
		df['normal'] = df['path'].map(lambda x: 1 if 'normal' in x else 0)
		# crop_coor, 얼굴 좌표 반환
		df['crop_coor'] = df.apply(lambda x: crop_coord(x['path']) if x['normal']==1 else None, axis=1)
		df = df.sort_values(by='name').fillna(method='backfill', limit=6).sort_index()
		
		
		# crop_image 저장할 경로 설정
		df['crop_path'] = df['path'].apply(lambda x: x.replace('images', args.crop_dir))
		# crop_image에 맞춘 csv 생성
		df_crop = df[['crop_path', 'label']].copy()
		df_crop.columns = ['path', 'label']
		df_crop['name'] = df_crop['path'].apply(lambda x: x.split('/')[-2]+'_'+x.split('/')[-1])
		# crop_image 생성 후 덮어씌우기
		df.apply(lambda x: crop_face(x['crop_coor'], x['path'], x['crop_path']), axis=1)


		copy_folder(dir_path, "ori_crop_images")
		for copy_path in df_crop['path']:
			new_path = copy_path.replace(args.crop_dir, "ori_crop_images")
			new_path = "/".join(new_path.split('/')[:-1])+'/crop_'+new_path.split('/')[-1]
			shutil.copy(copy_path, new_path)

		label_series = df['path'].map(lambda x: x.replace('images', 'ori_crop_images'))
		crop_series = df_crop['path'].map(lambda x: x.replace(args.crop_dir, 'ori_crop_images'))
		crop_series = crop_series.map(lambda x: "/".join(x.split('/')[:-1])+"/crop_"+x.split('/')[-1])
		df_merge = pd.DataFrame(label_series.append(crop_series), columns=['path'])
		df_merge['label'] = df['label'].append(df['label'])
		df_merge = df_merge.reset_index(drop=True)
		df_merge['name'] = df_merge['path'].map(lambda x: x.split('/')[-2]+'_'+x.split('/')[-1])
		

		# train_with_label에 crop_coord, crop_path 추가해서 재생성
		df.to_csv(label_path, index=False)
		df_crop.to_csv(dir_path+'/train_with_crop.csv', index=False)
		df_merge.to_csv(dir_path+'/train_with_all.csv', index=False)
