import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

import keras.backend as K
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from PIL import Image
from cv2 import cv2

size = (150, 150, 3)

def preprocess_img(image):
    test_data = []
    image = np.array(image)
    image = cv2.resize(image,(150,150))
    image = np.dstack([image, image, image])
    image = image.astype('float32') / 255
    print(np.size(image))
    test_data.append(image)
    print(np.size(test_data))
    return image


class Predictor: 
    def __init__(self):
        self.model = keras.models.load_model(r'C:\Users\jiang\OneDrive\Desktop\mais202_webapp\model\keras_model.h5')

    def predict(self, request):
        f = request.files['image']
        image = Image.open(f)
        image = preprocess_img(image)
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image)
        print(prediction)
        return prediction