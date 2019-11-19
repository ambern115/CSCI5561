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
    if (batch_size > len(im_train)):
        batch_size = len(im_train)
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
    y = np.dot(w,x) + b;

    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO

    dl_dw = np.empty((0,))  # 1 x (m*n)

    # Unsure if this is how x height should be accessed
    for i in range(len(x)):
        tmp = np.dot(w[i],x[i]) + b[i]
        if (tmp > 0):
            val = np.dot((tmp - y[i]), np.linalg.transpose(x[i]))
            dl_dw = np.append(dl_dw, [val], axis=0)
        else:
            zero = np.zeros((len(x),))
            dl_dw = np.append(dl_dw, [zero], axis=0)
    

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
    n_iters = 100
    learn_rate = 1
    decay_rate = 0.5  # (0,1]
    # Initialize wights with a Gaussian noise 
    w = np.random.normal(size=((10,196)))
    b = np.zeros((10,1))
    k = 1

    for i in range(n_iters):
        if (i % 1000 == 0):
            learn_rate = decay_rate*learn_rate
        dl_dw = np.zeros((1,10*196))
        dl_db = np.zeros((1,10))

        for img in range(len(mini_batch_x)):
            label_pred = fc(mini_batch_x[img], w, b)
            l, dl_dy = loss_euclidean(label_pred, mini_batch_y[img])
            dl_dx, tmp_dl_dw, tmp_dl_db = fc_backward(dl_dy, mini_batch_x[img], w, b, mini_batch_y[img])
            dl_dw = dl_dw + tmp_dl_dw
            dl_db = dl_db + tmp_dl_db
        if (k > len(mini_batch_x)):
            k = 1
        else:
            k += 1
        # Update weights and bias
        w = w - learn_rate * dl_dw
        b = b - learn_rate * dl_db
    
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



