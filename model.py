import tensorflow as tf


def get_model(input_shape, decay, output_node_size):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(input_layer)
    x = tf.keras.layers.Dense(units=128, kernel_initializer='he_normal', activation='relu')(x)
    x = tf.keras.layers.Dense(units=9, kernel_initializer='he_normal', activation='softmax', name='output')(x)
    return tf.keras.models.Model(input_layer, x)
