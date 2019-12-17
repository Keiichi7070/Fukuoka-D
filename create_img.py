import os
import keras
import cv2
import glob
from PIL import Image
from keras.utils import np_utils
from shutil import copy
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re

def list_pictures(directory, ext='jpg'):
    return [os.path.join(root, f)
        for root, _, files in os.walk(directory) for f in files
        if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

# モデルの読込
# 保存したjsonファイルとhdf5ファイルを読み込む。モデルを学習に使うにはcompileが必要。
from keras.models import model_from_json

# JSON形式のデータを読み込んでモデルとして復元。学習で使うにはまたコンパイルが必要なので注意。
with open('mnist.model', 'r') as f:
    json_string = f.read()
model = model_from_json(json_string)

# モデルにパラメータを読み込む。前回の学習状態を引き継げる。
model.load_weights('param.hdf5')
print('Loaded the model.')

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

Xj = []
Yj = []
num = []　#何番目の画像がぶれているかを保存
a = [1] #ぶれている画像との比較用
i = 0 #画像の番号

#画像フォルダ(img_file/test)の読み込み
for picture in list_pictures('img_file/test'): 
    img = img_to_array(load_img(picture))
    re_img = cv2.resize(img, (64, 64))
    Xj.append(re_img)
    Yj.append(1)
    
# arrayに変換
Xj = np.asarray(Xj)
Yj = np.asarray(Yj)
    
# 画素値を0から1の範囲に変換
Xj = Xj.astype('float32')
Xj = Xj / 255.0

# クラスの形式を変換
Yj = np_utils.to_categorical(Yj, 2)

# テストデータに適用
predict_classes = model.predict_classes(Xj)
for predict_test in predict_classes:
    if np.allclose(predict_test, a) == True:
        num.append(i)
    i += 1
# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(Yj, axis=1)})
# confusion matrix
pd.crosstab(mg_df['class'], mg_df['predict'])   

p = 0　#画像の番号
os.makedirs('image', exist_ok=True)　#imageというフォルダの作成
for path in glob.glob(os.path.join('img_file/test', '*.jpg')): #画像の読み込み
    if p in num: 
        copy(path,'{}/{}.jpg'.format('image',str(p).zfill(5))) #画像ファイルのコピー
        basename = os.path.basename('img_file/test') 
    p += 1