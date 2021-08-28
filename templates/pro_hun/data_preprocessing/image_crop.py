import cv2
import pandas as pd
import cvlib as cv
import os 
from pathlib import Path
from utils.util import ensure_dir
from PIL import Image
import tqdm

from utils.util import ensure_dir


def crop_coord(imagePath):
	image = cv2.imread(imagePath)
	face, confidence = cv.detect_face(image)
	H, W, C = image.shape
	x, y, w, h = face[0]

	return [x, y, w, h]


def crop_face(coord_list, path):
	x, y, w, h = coord_list
	image = cv2.imread(path)
	H, W, C = image.shape
	image_array = image[max(y-100,0):min(h+100,H), max(0,x-100):min(w+100,W)]
	image = Image.fromarray(image_array)

	crop_path = Path(path)
	ensure_dir(str(crop_path.parent))
	image.save(crop_path)
	

if __name__ == '__main__':
	label_path = '/opt/ml/image-classification-level1-12/templates/data/train/train_with_label.csv'
	df = pd.read_csv(label_path)
	df['normal'] = df['path'].map(lambda x: 1 if 'normal' in x else 0)
	df_detect = df[df['normal']==1]
	df['crop_coor'] = df.apply(lambda x: crop_coord(x['path']) if x['normal']==1 else None,axis=1)
	df = df.sort_values(by='name').fillna(method='backfill', limit=6).sort_index()
	df['crop_path'] = df['path'].apply(lambda x: x.replace('images', 'image_crop'))
	df.apply(lambda x: crop_face(x['crop_coor'], x['crop_path']), axis=1)
	df.to_csv(label_path, index=False)