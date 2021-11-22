import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, BatchNormalization, Add, Input, ReLU, DepthwiseConv2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications import MobileNetV2


def depth_block(x, strides):
    x = DepthwiseConv2D(3,strides=strides,padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def expansion_block(x, filters):
    x = Conv2D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def projection_block(x, filters):
    x = Conv2D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    return x


def combined(x, filters_1, filters_2, strides=1):
    x = expansion_block(x, filters_1)
    x = depth_block(x, strides)
    x = projection_block(x, filters_2)
    return x


def mobile_net(input_shape):
    input = tf.keras.Input(input_shape)
    x = Conv2D(16, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = combined(x, 16*6, 24)
    y = combined(x, 24*6, 24)
    x = Add()([x, y])
    
    x = combined(x, 24*6, 32, strides=2)
    for _ in range(2):
        y = combined(x, 32*6, 32)
        x = Add()([x, y])
        # 16x16x32

    x = combined(x, 32*6, 64, strides=2)
    y = combined(x, 64*6, 64)
    x = Add()([x, y])

    x = combined(x, 64*6, 128, strides=2)

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    return model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

datagen_train = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True)

datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)

datagen_train.fit(x_train)
datagen_test.fit(x_test)

model = mobile_net(input_shape=(32, 32, 3))
model.summary()

model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1),
              optimizer='adam',
              metrics=['accuracy'])

epochs = 50
hist = model.fit(datagen_train.flow(x_train, y_train, batch_size=64), epochs=epochs, 
                 validation_data=datagen_test.flow(x_test, y_test))

model.evaluate(datagen_test.flow(x_test, y_test))

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()