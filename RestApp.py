import os
import gc
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop,CenterCrop, RandomRotation

import cv2
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
#print("using {} device".format(device))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print('Device:', device)
#print('Current cuda device:', torch.cuda.current_device())
#print('Count of using GPUs:', torch.cuda.device_count())

categories = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
dir_c = './train_img_aug(150)/cafe_aug/'
dir_r = './train_img_aug(150)/restaurant_aug/'

def load_img(dir):                # 이미지를 불러온 뒤 X로 반환 , 해당 폴더명과 같은 라벨 Y 반환

    X = []
    Y = []

    for index , category in enumerate(categories):
        label = index + 1
        img_dir = dir + category +'/'

        for top, directory , f  in os.walk(img_dir):
            for filename in f:
                img = cv2.imread(img_dir + filename , cv2.IMREAD_COLOR)
                img = cv2.resize(img,(200,200), cv2.INTER_AREA) # 1440 x 1440 x 3은 너무 크기 때문에 360 x 360 x 3으로 변환
                #img_1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                Y.append(label)

    return X , Y

img_c ,label_c = load_img(dir_c) # 각각의 이미지를 폴더에서 불러움
img_r ,label_r = load_img(dir_r)
img_c = np.array(img_c)        #이미지 수정 및 병합을 위해 numpy형식의 배열로 바꿈
label_c = np.array(label_c)
img_r = np.array(img_r)
label_r = np.array(label_r)
label_r += 10           # cafe 와의 라벨 차이를 위해 restaurant는 +10 ( 총 0 ~19 번까지의 라벨 존재)
X = np.vstack([img_c,img_r]) # 카페와 식당의 이미지 병합
Y = np.hstack([label_c,label_r]) # 카페와 식당의 라벨 병합
X = np.array(X, dtype="float32") / 255.0    # 분류 정확도를 위한 정규화 ( 0 ~ 255 -> 0 ~ 1)
le = LabelBinarizer() # 원핫 인코딩
Y = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size = 0.20 ,stratify=Y, random_state=42) # 학습 / 테스트 스플릿
X_train , X_val , Y_train , Y_val = train_test_split(X_train,Y_train , test_size = 0.20 , stratify = Y_train , random_state = 42)
# 학습 / 검증 스플릿 ( 추후 실전 정확도는 test로 , 학습에 대한 검증은 validation으로 하기 위함)

print(X_train.shape)
print(X_val.shape)

from keras.preprocessing.image import ImageDataGenerator # 데이터의 추가 증강을 위한 테크닉
from skimage import io

aug = ImageDataGenerator(
rotation_range=10,
zoom_range=0.05,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.15,
horizontal_flip=False,
fill_mode="nearest")

# MODEL  설계
model = Sequential()
model.add(Conv2D(32, kernel_size = (5,5),
                activation = 'relu',
                padding = 'same',
                input_shape = (200, 200, 3)))
model.add(MaxPooling2D(pool_size = (2,2),
                      strides = (2,2)))
model.add(Conv2D(64, kernel_size = (5,5),
                activation = 'relu',
                padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(128, kernel_size = (5,5),
                activation = 'relu',
                padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(20, activation = 'softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy',
             optimizer = 'Adam', metrics = ['accuracy'])

hist = model.fit(X_train, Y_train, epochs = 10, validation_data = (X_val, Y_val), batch_size = 4)

y_hat = model.predict(X_test)

target = []
for i in range(len(y_hat)):
    target.append(np.argmax(y_hat[i]))

plt.imshow(X_test[0])
target[0]