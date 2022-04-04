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
            training_view=False,
            pretrained_model_path=''):
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.training_view_flag = training_view
        self.colors = [
            {'name': 'black', 'bgr': [0, 0, 0]},
            {'name': 'gray', 'bgr': [127, 127, 127]},
            {'name': 'white', 'bgr': [255, 255, 255]},
            {'name': 'red', 'bgr': [0, 0, 255]},
            {'name': 'orange', 'bgr': [0, 128, 255]},
            {'name': 'yellow', 'bgr': [0, 255, 255]},
            {'name': 'green', 'bgr': [0, 255, 0]},
            {'name': 'blue', 'bgr': [255, 0, 0]},
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
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.Dense(units=num_classes, kernel_initializer='glorot_normal', activation='softmax', name='output')(x)
        return tf.keras.models.Model(input_layer, x)

    def fit(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.momentum)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'])
        self.model.summary()

        break_flag = False
        iteration_count = 0
        while True:
            for batch_x, batch_y in self.train_data_generator.flow():
                logs = self.model.train_on_batch(batch_x, batch_y, return_dict=True)
                iteration_count += 1
                if self.training_view_flag:
                    # self.training_view.update(self.model)
                    pass
                print(f'\r[iteration count : {iteration_count:6d}] acc : {logs["acc"]:.4f}, loss => {logs["loss"]:.4f}', end='')
                if iteration_count % 5000 == 0:
                    self.model.save(f'checkpoints/model_{iteration_count}_iter.h5', include_optimizer=False)
                    print('\n')
                if iteration_count == self.iterations:
                    break_flag = True
                    break
            if break_flag:
                break

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
            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)
