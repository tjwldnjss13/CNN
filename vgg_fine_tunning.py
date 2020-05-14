
from keras import models, layers, optimizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

base_dir = 'C://DeepLearningData/dogs-vs-cats/mini'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    height_shift_range=0.2,
    width_shift_range=0.2,
    shear_range=0.1,
    rotation_range=30,
    horizontal_flip=True
)
train_batches = datagen.flow_from_directory(train_dir, target_size=(224, 224), classes=['dogs', 'cats'], batch_size=5)
valid_batches = datagen.flow_from_directory(valid_dir, target_size=(224, 224), classes=['dogs', 'cats'], batch_size=3)
test_batches = datagen.flow_from_directory(test_dir, target_size=(224, 224), classes=['dogs', 'cats'], batch_size=3)

test_images, test_labels = next(test_batches)
for i in range(3):
    plt.figure(i + 1)
    plt.imshow(test_images[i])

print(test_labels)
plt.show()
print(test_batches.class_indices)

vgg = VGG16()

model = models.Sequential()
for layer in vgg.layers:
    if layer.name is not 'predictions':
        layer.trainable = False
        model.add(layer)

model.add(layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['acc'])
model.fit(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=3, epochs=5, verbose=2)

predictions = model.predict_generator(test_batches, steps=1, verbose=2)
print(predictions)
