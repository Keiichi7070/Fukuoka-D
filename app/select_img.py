import os
import keras
import cv2
from keras.utils import np_utils
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


#グレースケール化
def gray(img):
    grayed_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #グレートスケール化
    grayed3_img = np.array([grayed_img, grayed_img, grayed_img]) #
    re_grayed3_img = np.reshape(grayed3_img, [64, 64, 3])
    
    return re_grayed3_img


# フォルダの中にある画像を順次読み込む
# カテゴリーは0から始める
X = []
Y = []
width = 64
height = 64

def mkli(img, i):
    X.append(img)
    Y.append(i)
        
# ぼやけていない画像
for picture in list_pictures('img_file/clear'):
    img = img_to_array(load_img(picture)) #PIL形式をndarray型に変換する
    re_img = cv2.resize(img, (64, 64)) #画像を縮小する
    mkli(re_img, 0)
    gray_img = gray(re_img)
    mkli(gray_img, 0)
    re_hflipped_img = cv2.flip(re_img, 1) 
    re_vflipped_img = cv2.flip(re_img, 0) 
    re_hvflipped_img = cv2.flip(re_vflipped_img, 1)
    mkli(re_hflipped_img, 0)
    mkli(re_vflipped_img, 0)
    mkli(re_hvflipped_img, 0)
    gray_hflipped_img = cv2.flip(gray_img, 1) 
    gray_vflipped_img = cv2.flip(gray_img, 0) 
    gray_hvflipped_img = cv2.flip(gray_vflipped_img, 1)
    mkli(gray_hflipped_img, 0)
    mkli(gray_vflipped_img, 0)
    mkli(gray_hvflipped_img, 0)
    center = (int(width/2),int(height/2))
#画像を10度ずつ回転させる
    a = 0
    while a < 360:
        a += 10;
        rotMat = cv2.getRotationMatrix2D(center, a, 1)
        rotated_image = cv2.warpAffine(re_img, rotMat, (width,height))
        mkli(rotated_image, 0)
        
    a = 0
    while a < 360:
        a += 10;
        rotMat = cv2.getRotationMatrix2D(center, a, 1)
        rotated_image = cv2.warpAffine(gray_img, rotMat, (width,height))
        mkli(rotated_image, 0)

# ぼやけている画像
for picture in list_pictures('img_file/blur'):
    img = img_to_array(load_img(picture)) #PIL形式をndarray型に変換する
    re_img = cv2.resize(img, (64, 64)) #画像を縮小する
    mkli(re_img, 1)
    gray_img = gray(re_img)
    mkli(gray_img, 1)
    re_hflipped_img = cv2.flip(re_img, 1) 
    re_vflipped_img = cv2.flip(re_img, 0) 
    re_hvflipped_img = cv2.flip(re_vflipped_img, 1)
    mkli(re_hflipped_img, 1)
    mkli(re_vflipped_img, 1)
    mkli(re_hvflipped_img, 1)
    gray_hflipped_img = cv2.flip(gray_img, 1) 
    gray_vflipped_img = cv2.flip(gray_img, 0) 
    gray_hvflipped_img = cv2.flip(gray_vflipped_img, 1)
    mkli(gray_hflipped_img, 1)
    mkli(gray_vflipped_img, 1)
    mkli(gray_hvflipped_img, 1)
    center = (int(width/2),int(height/2))
    
    a = 0
    while a <= 360:
        a += 10;
        rotMat = cv2.getRotationMatrix2D(center, a, 1)
        rotated_image = cv2.warpAffine(re_img, rotMat, (width,height))
        mkli(rotated_image, 1)
        
    a = 0
    while a <= 360:
        a += 10;
        rotMat = cv2.getRotationMatrix2D(center, a, 1)
        rotated_image = cv2.warpAffine(gray_img, rotMat, (width,height))
        mkli(rotated_image, 1)
    
# arrayに変換
X = np.asarray(X)
Y = np.asarray(Y)


# 画素値を0から1の範囲に変換
X = X.astype('float32')
X = X / 255.0

# クラスの形式を変換
Y = np_utils.to_categorical(Y, 2)

# 学習用データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)



# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))       # クラスは2個
model.add(Activation('softmax'))


# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])


# 実行。出力はなしで設定(verbose=0)
history = model.fit(X_train, y_train, batch_size=10, epochs=50,
                   validation_data = (X_test, y_test), verbose = 0)


#精度のグラフの表示
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()


## 損失関数のグラフ(1点)
plt.plot(history.history['loss'], label="loss for training")
plt.plot(history.history['val_loss'], label="loss for validation")
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()


# テストデータに適用
predict_classes = model.predict_classes(X_test)

# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(y_test, axis=1)})

# confusion matrix
pd.crosstab(mg_df['class'], mg_df['predict'])


# モデルの保存
# モデルはjson形式、パラメータはhdf5形式でそれぞれ保存

# モデルをJSON形式に変換。
json_string = model.to_json()
with open('select_img_best2.model', 'w') as f:
    f.write(json_string)
    
# モデルパラメータの書き出し
model.save_weights('param_best2.hdf5')
print('Saved the model.')