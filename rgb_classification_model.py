import os
import random
from glob import glob
from time import time

import cv2
import tensorflow as tf

from generator import RGBClassificationModelDataGenerator

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
live_view_previous_time = time()


class RGBClassificationModel:
    def __init__(
            self,
            input_shape,
            lr,
            momentum,
            batch_size,
            iterations,
            training_view=False,
            pretrained_model_path=''):
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.training_view_flag = training_view

        if pretrained_model_path == '':
            self.model = self.get_model(self.input_shape)
        else:
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)

        self.train_data_generator = RGBClassificationModelDataGenerator(
            input_shape=self.input_shape,
            batch_size=self.batch_size)

    def get_model(self, input_shape):
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(input_layer)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(x)
        x = tf.keras.layers.Dense(units=8, kernel_initializer='glorot_normal', activation='softmax', name='output')(x)
        return tf.keras.models.Model(input_layer, x)

    def fit(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.momentum)
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy())
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
                print(f'\r[iteration count : {iteration_count:6d}] loss => {logs["loss"]:.4f}', end='')
                if iteration_count % 5000 == 0:
                    self.model.save(f'checkpoints/model_{iteration_count}_iter.h5', include_optimizer=False)
                    print('\n')
                if iteration_count == self.iterations:
                    break_flag = True
                    break
            if break_flag:
                break

    def predict_rgb_data(self):
        pass
