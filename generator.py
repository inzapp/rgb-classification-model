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

    def augment(self, color):
        color_name = color['name']
        bgr = np.asarray(color['bgr']).reshape((1, 1, 3)).astype('uint8')
        h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape((3,)).astype('float32')
        if color_name == 'black' or color_name == 'gray' or color_name == 'white':
            pass  # do not augment hue and saturation value
        else:
            h += np.random.randint(-5, 6)
            s += np.random.randint(-100, 0)
        v += np.random.randint(-30, 31)
        h = np.clip(h, 0, 255)
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)
        hsv = np.asarray([h, s, v]).reshape((1, 1, 3)).astype('uint8')
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape((3,))

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        debug = False
        for _ in range(self.batch_size):
            color = self.colors[np.random.randint(len(self.colors))]
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
            x = np.asarray(augmentated_bgr).astype('float32') / 255.0
            y = np.zeros(shape=(len(self.colors),), dtype=np.float32)
            y[index] = 1.0
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size, 3)).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, len(self.colors))).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return 4096
