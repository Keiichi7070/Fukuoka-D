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

# フォルダの中にある画像を順次読み込む
# カテゴリーは0から始める

X = []
Y = []

# ぼやけていない画像
for picture in list_pictures('img_file/clear'):
    img = img_to_array(load_img(picture))　#PIL形式をndarray型に変換する
    re_img = cv2.resize(img, (64, 64))　#画像を縮小する
    X.append(re_img)
    Y.append(0)
#左右を反転させる
for picture in list_pictures('img_file/clear'):
    img = img_to_array(load_img(picture))
    re_img = cv2.resize(img, (64, 64))
    hflipped_re_img = cv2.flip(re_img, 1)
    X.append(hflipped_re_img)
    Y.append(0)
#上下を反転させる
for picture in list_pictures('img_file/clear'):
    img = img_to_array(load_img(picture))
    re_img = cv2.resize(img, (64, 64))
    vflipped_re_img = cv2.flip(re_img, 0)
    X.append(vflipped_re_img)
    Y.append(0)

# ぼやけている画像
for picture in list_pictures('img_file/blur'):
    img = img_to_array(load_img(picture))
    re_img = cv2.resize(img, (64, 64))
    X.append(re_img)
    Y.append(1)
#左右を反転させる
for picture in list_pictures('img_file/blur'):
    img = img_to_array(load_img(picture))
    re_img = cv2.resize(img, (64, 64))
    hflipped_re_img = cv2.flip(re_img, 1)
    X.append(hflipped_re_img)
    Y.append(1)
#上下を反転させる
for picture in list_pictures('img_file/blur'):
    img = img_to_array(load_img(picture))
    re_img = cv2.resize(img, (64, 64))
    vflipped_re_img = cv2.flip(re_img,0)
    X.append(vflipped_re_img)
    Y.append(1)

# arrayに変換
X = np.asarray(X)
Y = np.asarray(Y)

# 画素値を0から1の範囲に変換
X = X.astype('float32')
X = X / 255.0

# クラスの形式を変換
Y = np_utils.to_categorical(Y, 2)

# 学習用データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)

# CNNを構築
model = Sequential()

## ２次元畳み込み層１
model.add(Conv2D(filters = 32,
                 activation = 'relu',
                 kernel_size=(3, 3),
                 input_shape = (64, 64, 3)))
## ２次元畳み込み層２
model.add(Conv2D(filters = 64,
                 activation = 'relu',
                 kernel_size = (3, 3)))
## maxプーリング層
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
## 全結合層１
model.add(Flatten())
model.add(Dense(units = 128,
                activation = 'relu'))
model.add(Dropout(0.5))
## 全結合層２
model.add(Dense(units = 2,
                 activation = 'softmax'))



model.summary()

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# 実行。出力はなしで設定(verbose=0)。
history = model.fit(X_train, y_train, batch_size=5, epochs=200,
                   validation_data = (X_test, y_test), verbose = 0)

#精度のグラフの表示
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
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
with open('select_img.model', 'w') as f:
    f.write(json_string)
    
# モデルパラメータの書き出し
model.save_weights('param.hdf5')
print('Saved the model.')
