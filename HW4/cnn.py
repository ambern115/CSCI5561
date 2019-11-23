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
import scipy.stats as scistats  # Only used for initialize weight with normal distribution


def get_mini_batch(im_train, label_train, batch_size):
    # Randomly sample batch_size # samples from images
    mini_batch_x = np.empty((0,32,196))
    mini_batch_y = np.empty((0,32,10))

    if (batch_size*batch_size > len(im_train[0])):
        num_indices = len(im_train[0])
    else:
        num_indices = batch_size*batch_size
    indices = random.sample(range(len(im_train[0])), num_indices)
    idx_counter = 0

    # Find number of unique labels in label_train
    num_labels = len(Counter(np.ravel(label_train)).keys())

    for batch in range(batch_size):
        batch_x = np.empty((0,196))
        batch_y = np.empty((0,10))

        # Populate mini batches with data at sampled indices
        for i in range(batch_size):
            if (idx_counter < num_indices):

                img_col = np.array(im_train[:,indices[idx_counter]])
                # Manipulate the image data so that it is in a proper order
                # img_col = np.matrix.transpose(img_col.reshape(14,14)).reshape(196,)
                batch_x = np.append(batch_x, [img_col], axis=0)

                lbl_col = np.zeros((1,10))
                lbl_col[0][label_train[0][indices[idx_counter]]] = 1;
                batch_y = np.append(batch_y, lbl_col, axis=0)

                idx_counter += 1

        mini_batch_x = np.append(mini_batch_x, [batch_x], axis=0)
        mini_batch_y = np.append(mini_batch_y, [batch_y], axis=0)

    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    x = x.reshape(196,)
    y = np.dot(w,x) + b

    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dx = np.dot(dl_dy.reshape(1,10),w)

    x = x.reshape(196,1)
    dl_dw = np.dot(dl_dy.reshape(10,1),np.matrix.transpose(x))

    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l_tmp = np.linalg.norm(y - y_tilde)
    l = l_tmp*l_tmp

    dl_dy = -2*(y-y_tilde)

    dl_dy = dl_dy.reshape(1,10)

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
    n_iters = 100
    learn_rate = 0.005
    decay_rate = 0.5  # (0,1]

    # Initialize wights with a Gaussian noise with values b/w 0 and 1
    w = scistats.truncnorm(0.0, 1.0, loc=0.0, scale=1.0).rvs((10,196))
    b = np.zeros((10,))
    k = 0

    for i in range(n_iters):
        if (i % 1000 == 0):
            learn_rate = decay_rate*learn_rate
        dl_dw = np.zeros((10,196))
        dl_db = np.zeros((1,10))

        for img in range(len(mini_batch_x[0])):
            label_pred = fc(mini_batch_x[k][img], w, b)
            l, dl_dy = loss_euclidean(label_pred, mini_batch_y[k][img])
            dl_dx, tmp_dl_dw, tmp_dl_db = fc_backward(dl_dy, mini_batch_x[k][img], w, b, mini_batch_y[k][img])
            dl_dw = dl_dw + tmp_dl_dw
            dl_db = dl_db + tmp_dl_db

        if (k >= len(mini_batch_x)-1):
            k = 0
        else:
            k += 1
        # Update weights and bias
        w = w - learn_rate * dl_dw
        b = b - learn_rate * dl_db
    plt.imshow(w)
    plt.show()
    
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
    #main.main_slp()
    #main.main_mlp()
    #main.main_cnn()



