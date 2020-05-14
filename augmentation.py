from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

base_dir = 'C://DeepLearningData/dogs-vs-cats/train'
fn = os.path.join(base_dir, 'dog.132.jpg')

datagen = ImageDataGenerator(
    height_shift_range=0.2,
    width_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=20,
    horizontal_flip=True
)

image = np.expand_dims(Image.open(fn), 0)

aug_iter = datagen.flow(image)
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]

for i in range(10):
    plt.figure(i)
    plt.imshow(aug_images[i])

plt.show()