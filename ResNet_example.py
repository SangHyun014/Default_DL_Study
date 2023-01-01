#!/usr/bin/env python
# coding: utf-8

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
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.experimental import CosineDecay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50V2, ResNet50
from tensorflow.keras.layers.experimental.preprocessing import RandomCrop,CenterCrop, RandomRotation
from tensorflow.keras.applications import EfficientNetB0

import cv2
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using {} device".format(device))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

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

label_c.shape

label_r += 10           # cafe 와의 라벨 차이를 위해 restaurant는 +10 ( 총 0 ~19 번까지의 라벨 존재)

img_c.shape

X = np.vstack([img_c,img_r]) # 카페와 식당의 이미지 병합

Y = np.hstack([label_c,label_r]) # 카페와 식당의 라벨 병합

X.shape

X = np.array(X, dtype="float32") / 255.0    # 분류 정확도를 위한 정규화 ( 0 ~ 255 -> 0 ~ 1)

plt.imshow(X[0]) # 360 x 360 x 3으로 진행해도 이미지의 큰 수정은 없음

le = LabelBinarizer() # 원핫 인코딩
Y = le.fit_transform(Y)
print(Y.shape)
print(Y[:10])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y , test_size = 0.20 ,stratify=Y, random_state=42) # 학습 / 테스트 스플릿
X_train , X_val , Y_train , Y_val = train_test_split(X_train,Y_train , test_size = 0.20 , stratify = Y_train , random_state = 42)
# 학습 / 검증 스플릿 ( 추후 실전 정확도는 test로 , 학습에 대한 검증은 validation으로 하기 위함)

print(X_train.shape)
print(X_val.shape)

Y_train[:10]

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

batch_size = 4 # 32개의 배치로 학습
epochs = 50 # 50 에포크 진행

enet = ResNet50(
        input_shape=(200,200,3), # 입력 이미지와 맞춤
        weights=None,
        include_top=False
    )

model = tf.keras.Sequential([
        enet,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(20, activation='softmax') # 총 20개 분류이기 때문에 Dense(20)
    ])


#callbacks = [(monitor='val_accuracy' ,mode='max')]

early = EarlyStopping(monitor="val_accuracy", mode="max", patience=20, verbose = 1) # 20번 반복중 정확도가 최고치일때 early stop

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

history = model.fit(
    aug.flow(X_train, Y_train, batch_size=batch_size),
    epochs = epochs, 
    validation_data=(X_val, Y_val),
    verbose=1,
    callbacks=[early])