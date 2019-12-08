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
import scipy.signal as scisig # Only used to quickly convolve matrices in backward conv


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
    if (x.ndim == 2 and (len(x) < len(x[0]))):
        x = x.reshape(len(x[0]),)
    else:
        x = x.reshape(len(x),)
    y = np.dot(w,x) + b

    return y


def fc_backward(dl_dy, x, w, b, y):
    if (type(dl_dy) != np.float64):
        if (dl_dy.ndim == 2):
            dl_dy = dl_dy.reshape(len(dl_dy[0]),)
        dl_dx = np.dot(dl_dy.reshape(1,len(dl_dy)),w)
    else:
        dl_dx = np.dot(dl_dy,w)

    #x = x.reshape(196,1)
    #dl_dw = np.dot(dl_dy.reshape(10,1),np.matrix.transpose(x))
    if (x.ndim == 2):
        x = x.reshape(len(x[0]), 1)
    else:
        x = x.reshape(len(x),1)
    dl_dw = np.dot(dl_dy.reshape(len(dl_dy),1),np.matrix.transpose(x))

    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    l_tmp = np.linalg.norm(y - y_tilde)
    l = l_tmp*l_tmp

    dl_dy = -2*(y-y_tilde)

    dl_dy = dl_dy.reshape(1,10)

    return l, dl_dy


def loss_cross_entropy_softmax(x, y):
    if (x.ndim == 2):
        x = x.reshape(len(x[0]),)
    else:
        x = x.reshape(len(x),)
    max_x = np.max(x)  # Used to help prevent overflow

    ex_sum = 0.0
    for i in range(len(x)):
        ex_sum += math.exp(x[i]-max_x)

    l = 0.0
    for i in range(len(x)):
        l += y[i] * (x[i]-max_x - math.log(ex_sum))

    # Compute dl_dy w respect to x
    y_tilde = np.zeros((len(x),))  # len y = 10
    for i in range(len(x)):
        y_tilde[i] = math.exp(x[i]-max_x  - math.log(ex_sum))
    
    dl_dy = y_tilde-y

    return l, dl_dy

def relu(x):
    if (x.ndim == 2):
        x = x.reshape(len(x[0]),)
    y = np.zeros(x.shape)

    if (x.ndim == 3):
        y = np.zeros(x.shape)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if (x[i][j][0] > 0):
                    y[i][j][0] = x[i][j][0]
    else:
        for i in range(len(x)):
            if (x[i] > 0):
                y[i] = x[i]

    return y


def relu_backward(dl_dy, x, y):
    if (x.ndim == 2):
        x = x.reshape(len(x[0]),)
    if (y.ndim == 2):
        y = y.reshape(len(y[0]),)
    dl_dy = dl_dy.reshape(x.shape)

    dy_dx = np.zeros(dl_dy.shape)
    for i in range(len(x)):
        if (x[i] <= 0):
            dy_dx[i] = 0.0
        else:
            dy_dx[i] = 1.0

    dl_dx = np.multiply(dl_dy,dy_dx*y)
    
    return dl_dx

def uniform_zero_pad(matrix):
    # create new row of zeros for bottom and top
    # pads only by 1

    m_w = matrix.shape[1]
    m_h = matrix.shape[0]

    padded_m = np.zeros((m_h+2,m_w+2))

    for row in range(m_h):
        for elem in range(m_w):
            padded_m[row+1][elem+1] = matrix[row][elem]
    
    return padded_m



def conv(x, w_conv, b_conv):
    # TO DO
    w_h = w_conv.shape[2]
    w_w = w_conv.shape[3]
    c1 = w_conv.shape[1]
    c2 = w_conv.shape[0]

    x_w = x.shape[2]
    x_h = x.shape[1]

    # 0 pad each image layer in x (only 1 right now)
    # currently not scalable, only works for constant 1 padding on both x and y
    tmp_x = uniform_zero_pad(x[0])
    x = np.array([tmp_x])

    # flip all of the weight kernels to prepare for convolution
    for kernel in range(c2):
        w_conv[kernel][0] = np.flip(w_conv[kernel][0])
    
    # Begin weight convolution
    # For each layer C1
    for c1_layer in range(c1): # 1
        y = np.empty((0,x_h,x_w))
        for c2_layer in range(c2): # 3
            # Convolve x weight with corresponding x input
            conv_img = np.zeros((x_h,x_w))
            for i in range(x_h): # 14
                for j in range(x_w): # 14
                    # Multiply weights with x values
                    new_pixel = 0.0
                    for h in range(w_h): # 3
                        for w in range(w_w): # 3
                            # Access the correct layer/img of x
                            new_pixel += x[c1_layer][i+h][j+w]*w_conv[c2_layer][c1_layer][h][w]
                    # add b to each element in each matrix in each layer
                    conv_img[i][j] = new_pixel + b_conv[c2_layer]
            y = np.append(y, [conv_img], axis=0)
    
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # not correct... ran out of time
    dl_dw = scisig.convolve2d(x[0], dl_dy)
    dl_db = dl_dy

    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    x_w = x.shape[2]
    x_h = x.shape[1]

    new_w = math.floor(x_w/2.0)
    new_h = math.floor(x_h/2.0)

    y = np.zeros((new_h, new_w))

    for i in range(x_h-1):
        for j in range(x_w-1):
            mx = -10000
            for u in range(2):
                for v in range(2):
                    n = x[0][i+u][j+v]
                    if (n > mx):
                        mx = n
            y[i/2][j/2] = mx
            i += 1
            j += 1

    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def flattening(x):
    x_h = x.shape[0]
    x_w = x.shape[1]

    y = np.empty((x_w*x_h))

    for i in range(x_w):
        for j in range(x_h):
            y = np.append(y, x[j][i], axis=0)

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
    # plt.imshow(w)
    # plt.show()
    
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    n_iters = 100
    learn_rate = 0.05
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
            l, dl_dy = loss_cross_entropy_softmax(label_pred, mini_batch_y[k][img])
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
    # plt.imshow(w)
    # plt.show()
    
    return w, b


def train_mlp(mini_batch_x, mini_batch_y):
    n_iters = 400
    learn_rate = 0.0038
    decay_rate = 0.5  # (0,1]

    # Initialize wights with a Gaussian noise with values b/w 0 and 1
    w1 = scistats.truncnorm(0.0, 1.0, loc=0.0, scale=1.0).rvs((30,196))
    w2 = scistats.truncnorm(0.0, 1.0, loc=0.0, scale=1.0).rvs((10,30))
    b1 = np.zeros((30,))
    b2 = np.zeros((10,))
    k = 0

    for i in range(n_iters):
        if (i % 1000 == 0):
            learn_rate = decay_rate*learn_rate
        dl_dw1 = np.zeros((30,196))
        dl_dw2 = np.zeros((10,30))
        dl_db1 = np.zeros((1,30))
        dl_db2 = np.zeros((1,10))

        smallest_l = 100
        for img in range(len(mini_batch_x[0])):
            # Layer 1
            x1 = fc(mini_batch_x[k][img], w1, b1)
            y1 = relu(x1)
            # Layer 2
            y2 = fc(y1, w2, b2)
            # Soft max
            #l, dl_dy = loss_euclidean(y2, mini_batch_y[k][img])
            l, dl_dx = loss_cross_entropy_softmax(y2, mini_batch_y[k][img])
            
            # Back prop Layer 2
            #dl_dx = relu_backward(dl_dy, x2, y2)
            dl_dy, tmp_dl_dw2, tmp_dl_db2 = fc_backward(dl_dx, y1, w2, b2, mini_batch_y[k][img])
            # Back prop Layer 1
            dl_dx = relu_backward(dl_dy, x1, y1)
            dl_dy, tmp_dl_dw1, tmp_dl_db1 = fc_backward(dl_dx, mini_batch_x[k][img], w1, b1, mini_batch_y[k][img])

            # Update temp weights and biases
            dl_dw1 = dl_dw1 + tmp_dl_dw1
            dl_db1 = dl_db1 + tmp_dl_db1
            dl_dw2 = dl_dw2 + tmp_dl_dw2
            dl_db2 = dl_db2 + tmp_dl_db2

            if (abs(l) < smallest_l):
                smallest_l = l
        #print(smallest_l)
        if (k >= len(mini_batch_x)-1):
            k = 0
        else:
            k += 1
        # Update weights and biases
        w1 = w1 - learn_rate * dl_dw1
        b1 = b1 - learn_rate * dl_db1
        w2 = w2 - learn_rate * dl_dw2
        b2 = b2 - learn_rate * dl_db2
    # plt.imshow(w1)
    # plt.show()
    # plt.imshow(w2)
    # plt.show()
    
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    
    # Get x out of mini_batch_x
    x_1 = mini_batch_x[0][0].reshape(14,14)
    x = np.zeros((0, x_1.shape[0], x_1.shape[1]))
    x = np.append(x, [x_1], axis=0)

    # Initialize weights
    w_conv = scistats.truncnorm(0.0, 1.0, loc=0.0, scale=1.0).rvs((3,1,3,3))
    w_fc = scistats.truncnorm(0.0, 1.0, loc=0.0, scale=1.0).rvs((10,147))
    b_conv = np.zeros((3,))
    b_fc = np.zeros((10,1))

    #conv(x, w_conv, b_conv)


    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



