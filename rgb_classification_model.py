import os
import random
from glob import glob
from time import time

import cv2
import numpy as np
import tensorflow as tf

from generator import RGBClassificationModelDataGenerator

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
live_view_previous_time = time()


class RGBClassificationModel:
    def __init__(
            self,
            lr,
            momentum,
            batch_size,
            iterations,
            pretrained_model_path=''):
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.colors = [
            {'achromatic': True, 'name': 'black', 'bgr': [15, 15, 15]},
            {'achromatic': True, 'name': 'gray', 'bgr': [128, 128, 128]},
            {'achromatic': True, 'name': 'white', 'bgr': [245, 245, 245]},
            {'achromatic': False, 'name': 'red', 'bgr': [10, 10, 245]},
            {'achromatic': False, 'name': 'dark_red', 'bgr': [10, 10, 128]},
            {'achromatic': False, 'name': 'pastel_pink', 'bgr': [180, 180, 255]},
            # {'achromatic': False, 'name': 'orange', 'bgr': [10, 128, 245]},
            {'achromatic': False, 'name': 'brown', 'bgr': [10, 100, 180]},
            {'achromatic': False, 'name': 'pastel_peach', 'bgr': [165, 210, 255]},
            {'achromatic': False, 'name': 'yellow', 'bgr': [10, 245, 245]},
            {'achromatic': False, 'name': 'light_yellow', 'bgr': [128, 250, 250]},
            {'achromatic': False, 'name': 'military', 'bgr': [10, 128, 128]},
            {'achromatic': False, 'name': 'green', 'bgr': [10, 245, 10]},
            {'achromatic': False, 'name': 'dark_green', 'bgr': [10, 128, 10]},
            {'achromatic': False, 'name': 'pastel_green', 'bgr': [180, 255, 180]},
            {'achromatic': False, 'name': 'blue', 'bgr': [245, 10, 10]},
            {'achromatic': False, 'name': 'dark_blue', 'bgr': [128, 10, 10]},
            {'achromatic': False, 'name': 'pastel_blue', 'bgr': [255, 180, 180]},
            {'achromatic': False, 'name': 'aqua', 'bgr': [245, 245, 10]},
            {'achromatic': False, 'name': 'blue_green', 'bgr': [180, 180, 0]},
            {'achromatic': False, 'name': 'pink', 'bgr': [245, 10, 245]},
            {'achromatic': False, 'name': 'violet', 'bgr': [245, 10, 128]},
        ]

        if pretrained_model_path == '':
            self.model = self.get_model(num_classes=len(self.colors))
        else:
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)

        self.train_data_generator = RGBClassificationModelDataGenerator(
            batch_size=self.batch_size,
            colors=self.colors)

    def get_model(self, num_classes):
        input_layer = tf.keras.layers.Input(shape=(3,))
        x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(input_layer)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.Dense(units=num_classes, kernel_initializer='glorot_normal', activation='sigmoid', name='output')(x)
        return tf.keras.models.Model(input_layer, x)

    def fit(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.momentum)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['acc'])
        self.model.summary()

        iteration_count = 0
        while True:
            for batch_x, batch_y in self.train_data_generator.flow():
                logs = self.model.train_on_batch(batch_x, batch_y, return_dict=True)
                iteration_count += 1
                print(f'\r[iteration count : {iteration_count:6d}] acc : {logs["acc"]:.4f}, loss => {logs["loss"]:.4f}', end='')
                if iteration_count % 5000 == 0:
                    self.model.save(f'checkpoints/model_{iteration_count}_iter.h5', include_optimizer=False)
                    print('\n')
                if iteration_count == self.iterations:
                    print('train end successfully')
                    exit(0)

    def predict(self):
        while True:
            r = np.random.uniform(size=3)
            img = r.reshape((1, 1, 3)) * 255.0
            img = np.clip(img, 0.0, 255.0).astype('uint8')
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
            x = r.reshape((1, 3)).astype('float32')
            y = self.model.predict_on_batch(x=x).reshape((len(self.colors),))
            index = np.argmax(y)
            print(self.colors[index]['name'])

            bgr = color_img = self.colors[index]['bgr']
            color_img = np.asarray(bgr).reshape((1, 1, 3)).astype('uint8')
            color_img = cv2.resize(color_img, (128, 128), interpolation=cv2.INTER_NEAREST)
            res = np.concatenate((img, color_img), axis=1)
            cv2.imshow('img', res)
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)
