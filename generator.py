from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from cv2 import cv2


class RGBClassificationModelDataGenerator:
    def __init__(self, batch_size, colors):
        self.generator_flow = GeneratorFlow(batch_size, colors)

    def flow(self):
        return self.generator_flow


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, batch_size, colors):
        self.batch_size = batch_size
        self.colors = colors
        self.data_index = -1

    def augment(self, color):
        color_name = color['name']
        bgr = np.asarray(color['bgr'])
        if color['achromatic']:
            bgr = np.asarray(bgr).reshape((1, 1, 3)).astype('uint8')
            h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape((3,)).astype('float32')
            v += np.random.randint(-30, 31)
            hsv = np.clip(np.asarray([h, s, v]), 0.0, 255.0)
            hsv = hsv.reshape((1, 1, 3)).astype('uint8')
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape((3,))
        else:
            bgr = np.asarray(bgr).reshape((1, 1, 3)).astype('uint8')
            h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape((3,)).astype('float32')
            if color_name != 'yellow' and color_name != 'orange' and color_name != 'aqua':
                h += np.random.randint(-5, 6)
            if color_name.find('light') > -1 or color_name.find('pastel') > -1:
                v += np.random.randint(-10, 11)
            else:
                s += np.random.randint(-50, 51)
                v += np.random.randint(-50, 51)
            hsv = np.clip(np.asarray([h, s, v]), 0.0, 255.0)
            hsv = hsv.reshape((1, 1, 3)).astype('uint8')
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape((3,))
        bgr = bgr.reshape((3,)).astype('float32')
        for i in range(3):
            bgr[i] += np.random.randint(-5, 6)
        return np.clip(bgr, 0.0, 255.0).astype('uint8').reshape((3,))

    def next_data_index(self):
        self.data_index += 1
        if self.data_index == len(self.colors):
            self.data_index = 0
        return self.data_index

    def __getitem__(self, index):
        batch_x = []
        batch_y = []

        # self.batch_size = 100  # test
        debug = False
        for _ in range(self.batch_size):
            color = self.colors[self.next_data_index()]
            # color = self.colors[self.data_index]
            index = self.colors.index(color)
            augmentated_bgr = self.augment(color)
            if debug:
                bgr = np.asarray(color['bgr']).astype('uint8')
                bgr_img = cv2.resize(bgr.reshape(1, 1, 3), (128, 128), interpolation=cv2.INTER_NEAREST)
                a_bgr_img = cv2.resize(augmentated_bgr.reshape(1, 1, 3), (128, 128), interpolation=cv2.INTER_NEAREST)
                res = np.concatenate((bgr_img, a_bgr_img), axis=1)
                cv2.imshow('res', res)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)
            x = np.asarray(augmentated_bgr).astype('float32')
            y = np.zeros(shape=(len(self.colors),), dtype=np.float32)
            y[index] = 1.0
            batch_x.append(x)
            batch_y.append(y)
        # self.data_index += 1  # test
        batch_x = np.asarray(batch_x).reshape((self.batch_size, 3)).astype('float32') / 255.0
        batch_y = np.asarray(batch_y).reshape((self.batch_size, len(self.colors))).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return 4096
