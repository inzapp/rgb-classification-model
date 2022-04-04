from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from cv2 import cv2


class RGBClassificationModelDataGenerator:
    def __init__(self, input_shape, batch_size):
        self.generator_flow = GeneratorFlow(input_shape, batch_size)

    def flow(self):
        return self.generator_flow


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.colors = [
            [0, 0, 0],  # black
            [127, 127, 127],  # gray
            [255, 255, 255],  # white
            [0, 0, 255],  # red
            [0, 128, 255],  # orange
            [0, 255, 255],  # yellow
            [0, 255, 0],  # green
            [255, 0, 0],  # blue
        ]

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        debug = True
        for _ in range(self.batch_size):
            bgr = self.colors[np.random.randint(len(self.colors))]
            index = self.colors.index(bgr)
            bgr = np.asarray(bgr).reshape((1, 1, 3)).astype('uint8')
            h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape((3,)).astype('float32')
            h += np.random.randint(-5, 6)
            s += np.random.randint(-50, 51)
            v += np.random.randint(-10, 11)
            h = np.clip(h, 0, 255)
            s = np.clip(s, 0, 255)
            v = np.clip(v, 0, 255)
            hsv = np.asarray([h, s, v]).reshape((1, 1, 3)).astype('uint8')
            augmentated_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape((3,))
            if debug:
                bgr_img = cv2.resize(bgr.reshape(1, 1, 3), (128, 128), interpolation=cv2.INTER_NEAREST)
                a_bgr_img = cv2.resize(augmentated_bgr.reshape(1, 1, 3), (128, 128), interpolation=cv2.INTER_NEAREST)
                res = np.concatenate((bgr_img, a_bgr_img), axis=1)
                cv2.imshow('res', res)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)
            x = np.asarray(augmentated_bgr).astype('float32') / 255.0
            y = np.zeros(shape=(len(self.colors),), dtype=np.float32)
            y[index] = 1.0
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, len(self.colors))).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(1024 / self.batch_size))
