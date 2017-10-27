# -*- coding:utf-8 -*-
import random
import numpy as np
from collections import defaultdict
from keras.datasets import cifar10

class_num = 10
image_size = 32
img_channels = 3


def prepare_data(n):
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    train_data, test_data = color_preprocessing(train_data, test_data) # pre-processing

    criteria = n//10
    input_dict, input_x, input_y, classify_x, classify_y = defaultdict(int), list(), list(), list(), list()

    for image, label in zip(train_data,train_labels) :
        if input_dict[int(label)] != criteria :
            input_dict[int(label)] += 1
            input_x.append(image)
            input_y.append(label)

        classify_x.append(image)
        classify_y.append(label)


    input_x = np.asarray(input_x)
    input_y = np.asarray(input_y)
    classify_x = np.asarray(classify_x)
    classify_y = np.asarray(classify_y)

    print("Input data:", np.shape(input_x), np.shape(input_y))
    print("Classify data :", np.shape(classify_x), np.shape(classify_y))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(input_x))
    input_x = input_x[indices]
    input_y = input_y[indices]

    indices = np.random.permutation(len(classify_x))
    classify_x = classify_x[indices]
    classify_y = classify_y[indices]

    print("======Prepare Finished======")


    input_y_vec = np.zeros((len(input_y), 10), dtype=np.float)
    for i, label in enumerate(input_y) :
        input_y_vec[i, input_y[i]] = 1.0

    classify_y_vec = np.zeros((len(classify_y), 10), dtype=np.float)
    for i, label in enumerate(classify_y) :
        classify_y_vec[i, classify_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), 10), dtype=np.float)
    for i, label in enumerate(test_labels) :
        test_labels_vec[i, test_labels[i]] = 1.0


    return input_x, input_y_vec, classify_x, classify_y_vec, test_data, test_labels_vec


# ========================================================== #
# ├─ _random_crop()
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# └─ color_preprocessing()
# ========================================================== #

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch