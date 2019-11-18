import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main
import random
from collections import Counter  # Used in get_mini_batch


def get_mini_batch(im_train, label_train, batch_size):
    # Randomly sample batch_size # samples from images
    indices = random.sample(range(len(im_train)), batch_size)

    # Find number of unique labels in label_train
    num_labels = len(Counter(np.ravel(label_train)).keys())

    mini_batch_x = np.empty((0,196))
    mini_batch_y = np.empty((0,10))

    # Populate mini batches with data at sampled indices
    for i in indices:
        img_col = np.array(im_train[:,i])
        mini_batch_x = np.append(mini_batch_x, [img_col], axis=0)

        lbl_col = np.zeros((1,10))
        lbl_col[0][label_train[0][i]] = 1;
        mini_batch_y = np.append(mini_batch_y, lbl_col, axis=0)
        
    # Shape 196 x batch_size
    mini_batch_x = mini_batch_x.reshape((196,batch_size))
    # Shape 10 x batch_size
    mini_batch_y = mini_batch_y.reshape((num_labels,batch_size))

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    # TO DO
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    return l, dl_dy

def relu(x):
    # TO DO
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def flattening(x):
    # TO DO
    return y


def flattening_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



