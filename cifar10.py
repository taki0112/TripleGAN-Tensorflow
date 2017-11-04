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
    input_dict, labelled_x, labelled_y, unlabelled_x, unlabelled_y = defaultdict(int), list(), list(), list(), list()

    for image, label in zip(train_data,train_labels) :
        if input_dict[int(label)] != criteria :
            input_dict[int(label)] += 1
            labelled_x.append(image)
            labelled_y.append(label)

        unlabelled_x.append(image)
        unlabelled_y.append(label)


    labelled_x = np.asarray(labelled_x)
    labelled_y = np.asarray(labelled_y)
    unlabelled_x = np.asarray(unlabelled_x)
    unlabelled_y = np.asarray(unlabelled_y)

    print("labelled data:", np.shape(labelled_x), np.shape(labelled_y))
    print("unlabelled data :", np.shape(unlabelled_x), np.shape(unlabelled_y))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(labelled_x))
    labelled_x = labelled_x[indices]
    labelled_y = labelled_y[indices]

    indices = np.random.permutation(len(unlabelled_x))
    unlabelled_x = unlabelled_x[indices]
    unlabelled_y = unlabelled_y[indices]

    print("======Prepare Finished======")


    labelled_y_vec = np.zeros((len(labelled_y), 10), dtype=np.float)
    for i, label in enumerate(labelled_y) :
        labelled_y_vec[i, labelled_y[i]] = 1.0

    unlabelled_y_vec = np.zeros((len(unlabelled_y), 10), dtype=np.float)
    for i, label in enumerate(unlabelled_y) :
        unlabelled_y_vec[i, unlabelled_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), 10), dtype=np.float)
    for i, label in enumerate(test_labels) :
        test_labels_vec[i, test_labels[i]] = 1.0


    return labelled_x, labelled_y_vec, unlabelled_x, unlabelled_y_vec, test_data, test_labels_vec


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
    """
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])
    """
    x_train = x_train/127.5 - 1
    x_test = x_test/127.5 - 1
    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch