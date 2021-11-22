#%%
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model, Model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import cosine, euclidean


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True)

datagen_train.fit(x_train)


def extract_features(img):
    img = np.expand_dims(img, 0)
    img = datagen_train.flow(img, batch_size=1).next()
    ft = model.predict(img)[0]
    return ft


model = load_model('model_cifarv4')
model = Model(model.input, model.get_layer('global_average_pooling2d_1').output)

features = []
for img in x_train:
    feature = extract_features(img)
    features.append(feature)
features = np.array(features)

dir = 'image_retrieval'
images = []
for path in os.listdir(dir):
    image = cv2.imread(os.path.join(dir, path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
images = np.array(images)

axes = []
fig = plt.figure(figsize=(10, 10))
for i in range(len(images)):
    feature = extract_features(images[i])
    dist = [cosine(features[i], feature) for i in range(len(features))]
    #dist = [np.linalg.norm(features[i] - feature) for i in range(len(features))]
    ids = np.argsort(dist)[:5]
    for j in range(len(ids) + 1):
        axes.append(fig.add_subplot(10, len(ids) + 1, 6 * i + j + 1))
        if j == 0:
            plt.axis('off')
            plt.imshow(images[i])
        else:
            id = ids[j-1]
            plt.axis('off')
            plt.imshow(x_train[id])

plt.savefig('images_retrieved_cosine.png')
plt.show()
