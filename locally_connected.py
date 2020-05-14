import numpy as np
from PIL import Image
import os

def preprocessed_image(image_file):
    image = np.array(image_file)
    image_filtered = np.zeros((16 - 3 + 1, 16 - 3 + 1))
    for i in range(14):
        for j in range(14):
            image_filtered[i, j] = np.sum(image[i:i+3, j:j+3] * SOBEL_X)
    image_filtered = image_filtered.astype('float32') / 255.

    # Padding
    image_padded = np.zeros((17, 17))
    image_padded[1:15, 1:15] = image_filtered

    return image_padded


def train():
    pass


def cost_function(target, output):
    return np.sum(-target * np.log(output) - (1 - target) * np.log(1 - output)) / len(target)


def bit_cost_function(target, output):
    return -target * np.log(output) - (1 - target) * np.log(1 - output)


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def sigmoid_dif(x):
    return sigmoid(x) * (1 - sigmoid(x))


def leaky_relu(x):
    return np.maximum(.01 * x, x)


def leaky_relu_dif(x):
    if x > 0:
        return 1
    else:
        return .01


def relu(x):
    return np.maximum(0, x)


def relu_dif(x):
    if x > 0:
        return 1
    else:
        return 0


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


SOBEL_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
SOBEL_Y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

LABELS = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])


# Preprocessing image data
train_images = []
train_labels = []
data_path = '../digit data'

for n in range(10):
    for label in range(10):
        fn = str(label) + '.' + str(n) + '.png'
        image_path = os.path.join(data_path, fn)
        image = Image.open(image_path).convert('L')
        train_image = preprocessed_image(image)

        train_images.append(train_image)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)


# Define parameters
LR = .05
epochs = 50

W_ItoH1 = np.random.uniform(-.01, .01, (3, 3))
W_H1toH2 = np.random.uniform(-.01, .01, (5, 5))
W_H2toO = np.random.uniform(-.01, .01, (16, 10))

activation = leaky_relu
activation_dif = leaky_relu_dif

# Train
for epoch in range(epochs):
    print('[{:2d} / {} epoch]'.format(epoch+1, epochs), end='    ')
    epoch_loss = 0
    for data in range(100):
        image = train_images[data]
        label = train_labels[data]

        # Feedforward
        pre_H1 = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                pre_H1[i, j] = np.sum(image[2*i:2*i+3, 2*j:2*j+3] * W_ItoH1)
        post_H1 = activation(pre_H1)

        pre_H2 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                pre_H2[i, j] = np.sum(post_H1[i:i+5, j:j+5] * W_H1toH2)
        post_H2 = activation(pre_H2)

        post_H2_flat = np.reshape(post_H2, 16)
        pre_O = np.zeros(10)
        for i in range(10):
            pre_O[i] = np.sum(post_H2_flat * W_H2toO[:, i])
        post_O = sigmoid(pre_O)

        ce = bit_cost_function(LABELS[label], post_O)
        ce_loss = cost_function(LABELS[label], post_O)
        epoch_loss += ce_loss

        # Backpropagate
        D_post_O = -.5 * (LABELS[label] / post_O - (1 - LABELS[label]) / (1 - post_O))
        D_pre_O = np.zeros(10)
        for i in range(10):
            D_pre_O[i] = D_post_O[i] * sigmoid_dif(pre_O[i])

        W_H2toO_old = W_H2toO
        for i in range(16):
            for j in range(10):
                W_H2toO[i, j] -= LR * D_pre_O[j] * post_H2_flat[i]

        D_post_H2_flat = np.zeros(16)
        for i in range(16):
            for j in range(10):
                D_post_H2_flat[i] = W_H2toO_old[i, j] * D_pre_O[j]
        D_post_H2 = np.reshape(D_post_H2_flat, (4, 4))
        D_pre_H2 = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                D_pre_H2[i, j] = D_post_H2[i, j] * activation_dif(pre_H2[i, j])

        W_H1toH2_old = W_H1toH2
        for i in range(5):
            for j in range(5):
                W_H1toH2[i, j] -= LR * np.sum(post_H1[i:i+4, j:j+4] * D_pre_H2)

        W_H1toH2_old_inv = np.flip(W_H1toH2_old)
        D_pre_H2_padded = np.zeros((12, 12))
        D_pre_H2_padded[4:8, 4:8] = D_pre_H2
        for i in range(4, 8):
            for j in range(4, 8):
                D_pre_H2_padded[i][j] = D_pre_H2[i - 4][j - 4]
        D_post_H1 = np.zeros((8, 8))
        D_pre_H1 = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                D_post_H1[i, j] = np.sum(D_pre_H2_padded[i:i+5, j:j+5] * W_H1toH2_old_inv)
                D_pre_H1[i, j] = D_post_H1[i, j] * activation_dif(pre_H1[i, j])

        for i in range(3):
            for j in range(3):
                W_ItoH1[i, j] -= LR * np.sum(image[i:i+15:2, j:j+15:2] * D_pre_H1)

    epoch_loss /= 100
    print('Loss: {}'.format(epoch_loss))


# Test
test_image_path = os.path.join(data_path, '7.6.png')
test_image = preprocessed_image(Image.open(test_image_path).convert('L'))

pre_H1 = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        pre_H1[i, j] = np.sum(test_image[2*i:2*i+3, 2*j:2*j+3] * W_ItoH1)
post_H1 = activation(pre_H1)

pre_H2 = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        pre_H2[i, j] = np.sum(post_H1[i:i+5, j:j+5] * W_H1toH2)
post_H2 = activation(pre_H2)

post_H2_flat = np.reshape(post_H2, 16)
pre_O = np.zeros(10)
post_O = pre_O
for i in range(10):
    pre_O[i] = np.sum(post_H2_flat * W_H2toO[:, i])
post_O = activation(pre_O)
print(post_O)