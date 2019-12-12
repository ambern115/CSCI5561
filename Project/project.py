# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath
from skimage.exposure import rescale_intensity
import scipy.stats as scistats  # Only used for initialize weight with normal distribution


# Implement convolutional neural network (get basics working)
# Need to retrieve content reconstructions and style reconstructions from CNN 
# a balance between these will produce a styalized image filter

def filter_image(im, filtr, bias):
    print("Filtering Image")
    #im = [[1,2,4,5,5,0],[6,1,0,1,0,2],[0,0,1,1,0,1],[8,0,2,4,9,7],[2,3,1,0,8,7],[2,2,2,2,2,2]]
    if (len(im) == 0 or len(im[0]) == 0):
        print("Error: image to be filtered is too small.")
        return im
    if (len(filtr) == 0 or len(filtr[0]) == 0):
        print("Error: filter size too small.")
        return im

    im = im[0]
    filtr = filtr[0]
    bias = bias[0]

    # Reverse filter
    filtr = np.flip(filtr)

    # Below: perform the mulitplication and summing operations done in convolutions

    # Variables used in the loops below
    offset = math.floor(len(filtr)/2)
    im_filtered = []
    im_w = len(im)
    im_h = len(im[0])
    
    # Loop: convolve image and filter
    for row in range(im_w):
        row_filtered = []
        for col in range(im_h):
            # Multiply numbers with filter centered around selected pixel and add
            row_offset = offset+1
            temp_sum = 0
            for f_row in filtr:
                col_offset = offset
                row_offset -= 1
                for n in f_row:
                    if (n != 0):
                        temp_row = row-row_offset
                        temp_col = col-col_offset

                        # Check if pixel is outside im border
                        if (temp_row >= 0 and temp_col >= 0):
                            if (temp_row < im_w and temp_col < im_h):
                                temp_sum += im[temp_row][temp_col]*n

                    col_offset -= 1
            row_filtered.append(temp_sum + bias)
        im_filtered.append(row_filtered)

    #im_filtered = scipy.signal.convolve2d(im, filtr)

    return im_filtered

def convolve(image, kernel, bias):
  # get spatial dim of image and kernel
  (iH, iW) = image.shape[:2]
  (kH, kW) = kernel.shape[:2]

  # allocate memory for output image and pad
  pad = (kW - 1) // 2
  image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
  output = np.zeros((iH, iW), dtype="float32")

  # loop over input image, sliding kernel
  for y in np.arange(pad, iH + pad):
    for x in np.arange(pad, iW + pad):
      # extract ROI of image
      roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

      # peform actual convolution
      k = (roi * kernel).sum() 

      # store convolved val in ouput coord of output image
      output[y - pad, x - pad] = k + bias

  # rescale output image to be in range [0, 255]
  output = rescale_intensity(output, in_range=(0, 255))
  output = (output * 255).astype("uint8")

  # return output image
  return output

def convolution(image, filt, bias, s=1):
  '''
  Confolves `filt` over `image` using stride `s`
  '''
  (n_f, n_c_f, f_h, f_w) = filt.shape # filter dimensions

  # zero padding assumes filter is square
  # pad = (f_h - 1) // 2
  # image = np.array([cv2.copyMakeBorder(image[0], pad, pad, pad, pad, cv2.BORDER_REPLICATE)])
  n_c, in_dim_h, in_dim_w = image.shape # image dimensions
  
  out_dim_h = int((in_dim_h - f_h)/s)+1 # calculate output dimensions
  out_dim_w = int((in_dim_w - f_w)/s)+1 # calculate output dimensions

  # ensure that the filter dimensions match the dimensions of the input image
  assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"


  out = np.zeros((n_f,out_dim_h,out_dim_w)) # create the matrix to hold the values of the convolution operation
  # out = np.zeros((n_f, in_dim_h, in_dim_w))

  # convolve each filter over the image
  for curr_f in range(n_f):
    print(curr_f)

    curr_y = out_y = 0
    # move filter vertically across the image
    while curr_y + f_h <= in_dim_h:
        curr_x = out_x = 0
        # move filter horizontally across the image 
        while curr_x + f_w <= in_dim_w:
            # perform the convolution operation and add the bias
            out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f_h, curr_x:curr_x+f_w]) + bias[curr_f]
            curr_x += s
            out_x += 1
        curr_y += s
        out_y += 1
      
  return out

# must be avg pool...
def maxpool(image, f=2, s=2):
    # Downsample input `image` using a kernel size of `f` and a stride of `s`
    n_c, h_prev, w_prev = image.shape
    
    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - f)/s)+1 
    w = int((w_prev - f)/s)+1
    
    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((n_c, h, w)) 
    
    # slide the window over every part of the image using stride s. Take the maximum value at each step.
    for i in range(n_c):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + f <= w_prev:
                # choose the maximum value within the window at each step and store it to the output matrix
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled
    
def softmax(raw_preds):
    '''
    pass raw predictions through softmax activation function
    '''
    out = np.exp(raw_preds) # exponentiate vector of raw predictions
    return out/np.sum(out) # divide the exponentiated vector by its sum. All values in the output sum to 1.

def categoricalCrossEntropy(probs, label):
    '''
    calculate the categorical cross-entropy loss of the predictions
    '''
    return -np.sum(label * np.log(probs)) # Multiply the desired output label by the log of the prediction, then sum all values in the vector


def initializeFilter(size, scale = 1.0):
    '''
    Initialize filter using a normal distribution with and a 
    standard deviation inversely proportional the square root of the number of units
    '''
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    '''
    Initialize weights with a random normal distribution
    '''
    return np.random.standard_normal(size=size) * 0.01

def convolutionBackward(dconv_prev, conv_in, filt, s):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (n_f, n_c, f_h, f_w) = filt.shape
    (_, orig_dim_h, orig_dim_w) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f,1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f_h <= orig_dim_h:
            curr_x = out_x = 0
            while curr_x + f_w <= orig_dim_w:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f_h, curr_x:curr_x+f_w]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f_h, curr_x:curr_x+f_w] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    
    return dout, dfilt, dbias

# need to change to average pooling
def nanargmax(arr):
    '''
    return index of the largest non-nan value in the array. Output is an ordered pair tuple
    '''
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs 

# need to change to average pooling
def maxpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    '''
    (n_c, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout

def getWhiteNoiseImg(img_shape):
  out = scistats.truncnorm(0.0, 1.0, loc=0.0, scale=1.0).rvs(img_shape)
  out = out*255.0

  return out

def visualizeContent(image, filters, bias, response, conv_s):
  # Generate white noise image
  _, h, w = image.shape
  x = getWhiteNoiseImg((1, h, w))
  plt.imshow(x[0])
  plt.show()

  for i in range(30):
    # Get x's response
    F = convolution(x, filters, bias, conv_s) # convolution operation
    F[F<=0] = 0 # pass through ReLU non-linearity

    # Compute loss
    tmp = np.square(np.subtract(F, response))
    loss = 0.5 * tmp.sum()

    # Compute dL/dF
    dl_df = np.zeros(F.shape)
    for i in range(len(F)):
      for j in range(len(F[0])):
        for k in range(len(F[0][0])):
          if (F[i][j][k] > 0):
            dl_df[i][j][k] = F[i][j][k] - response[i][j][k]

    # Compute derivatives
    dx, df, db = convolutionBackward(dl_df, x, filters, conv_s)
    
    # Update x
    x = x - dx
    plt.imshow(x[0])
    plt.show()

  plt.imshow(x[0])
  plt.show()
  
  return x


def conv(image, params, conv_s, pool_f, pool_s):
    
    [f1, f2, w3, w4, b1, b2, b3, b4] = params 
    
    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 # pass through ReLU non-linearity
    visualizeContent(image, f1, b1, conv1, conv_s)
    
    print("Passed vis content")

    conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity
    
    pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation
    
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity
    
    out = w4.dot(z) + b4 # second dense layer
     
    probs = softmax(out) # predict class probabilities with the softmax activation function
    
    ################################################
    #################### Loss ######################
    ################################################
    
    loss = 1.0#categoricalCrossEntropy(probs, label) # categorical cross-entropy loss
        
    # ################################################
    # ############# Backward Operation ###############
    # ################################################

    # dout = probs - label # derivative of loss w.r.t. final dense layer output
    # dw4 = dout.dot(z.T) # loss gradient of final dense layer weights
    # db4 = np.sum(dout, axis = 1).reshape(b4.shape) # loss gradient of final dense layer biases
    
    # dz = w4.T.dot(dout) # loss gradient of first dense layer outputs 
    # dz[z<=0] = 0 # backpropagate through ReLU 
    # dw3 = dz.dot(fc.T)
    # db3 = np.sum(dz, axis = 1).reshape(b3.shape)
    
    # dfc = w3.T.dot(dz) # loss gradients of fully-connected layer (pooling layer)
    # dpool = dfc.reshape(pooled.shape) # reshape fully connected into dimensions of pooling layer
    
    # dconv2 = maxpoolBackward(dpool, conv2, pool_f, pool_s) # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
    # dconv2[conv2<=0] = 0 # backpropagate through ReLU
    
    # dconv1, df2, db2 = convolutionBackward(dconv2, conv1, f2, conv_s) # backpropagate previous gradient through second convolutional layer.
    # dconv1[conv1<=0] = 0 # backpropagate through ReLU
    
    # dimage, df1, db1 = convolutionBackward(dconv1, image, f1, conv_s) # backpropagate previous gradient through first convolutional layer.
    
    # grads = [df1, df2, dw3, dw4, db1, db2, db3, db4] 
    
    # return grads, loss
    return loss

def adamWhiteGD(content_img, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    update the parameters through Adam gradient descnet.
    to find image content representation
    '''
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    #X = batch[:,0:-1] # get batch inputs
    #X = X.reshape(len(batch), n_c, dim, dim)
    #Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    #batch_size = len(batch)
    
    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    
    #for i in range(batch_size):
    y = content_img
    #x = X[i]
    #y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
    
    # Collect Gradients for training example
    grads, loss = conv(x, y, params, 1, 2, 2)
    [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads
    
    df1+=df1_
    db1+=db1_
    df2+=df2_
    db2+=db2_
    dw3+=dw3_
    db3+=db3_
    dw4+=dw4_
    db4+=db4_

    cost_+= loss

    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
    
    bv1 = beta1*bv1 + (1-beta1)*db1
    bs1 = beta2*bs1 + (1-beta2)*(db1)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
   
    v2 = beta1*v2 + (1-beta1)*df2
    s2 = beta2*s2 + (1-beta2)*(df2)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                       
    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)
    
    v3 = beta1*v3 + (1-beta1) * dw3/batch_size
    s3 = beta2*s3 + (1-beta2)*(dw3/batch_size)**2
    w3 -= lr * v3/np.sqrt(s3+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v4 = beta1*v4 + (1-beta1) * dw4/batch_size
    s4 = beta2*s4 + (1-beta2)*(dw4/batch_size)**2
    w4 -= lr * v4 / np.sqrt(s4+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)
    

    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]
    
    return params, cost



# Alternative to steepest descent using gradient....
def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, w1, w2, b1, b2, b3, b4] = params
    
    X = batch[:,0:-1] # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1] # get batch labels
    
    cost_ = 0
    batch_size = len(batch)
    
    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    df3 = np.zeros(f3.shape)
    df4 = np.zeros(f4.shape)
    df5 = np.zeros(f5.shape)
    df6 = np.zeros(f6.shape)
    df7 = np.zeros(f7.shape)
    df8 = np.zeros(f8.shape)
    df9 = np.zeros(f9.shape)
    df10 = np.zeros(f10.shape)
    df11 = np.zeros(f11.shape)
    df12 = np.zeros(f12.shape)
    df13 = np.zeros(f13.shape)

    dw1 = np.zeros(w3.shape)
    dw2 = np.zeros(w4.shape)

    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)
    
    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(f3.shape)
    v4 = np.zeros(f4.shape)
    v5 = np.zeros(f5.shape)
    v6 = np.zeros(f6.shape)
    v7 = np.zeros(f7.shape)
    v8 = np.zeros(f8.shape)
    v9 = np.zeros(f9.shape)
    v10 = np.zeros(f10.shape)
    v11 = np.zeros(f11.shape)
    v12 = np.zeros(f12.shape)
    v13 = np.zeros(f13.shape)

    v14 = np.zeros(w1.shape)
    v15 = np.zeros(w2.shape)

    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)
    
    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(f3.shape)
    s4 = np.zeros(f4.shape)
    s5 = np.zeros(f5.shape)
    s6 = np.zeros(f6.shape)
    s7 = np.zeros(f7.shape)
    s8 = np.zeros(f8.shape)
    s9 = np.zeros(f9.shape)
    s10 = np.zeros(f10.shape)
    s11 = np.zeros(f11.shape)
    s12 = np.zeros(f12.shape)
    s13 = np.zeros(f13.shape)

    s14 = np.zeros(w1.shape)
    s15 = np.zeros(w2.shape)

    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)
    
    for i in range(batch_size): 
      x = X[i]
      y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1) # convert label to one-hot
      
      # Collect Gradients for training example
      grads, loss = conv(x, y, params, 1, 2, 2)
      [df1_, df2_, df3_, df4_, dw1_, dw2_, db1_, db2_, db3_, db4_] = grads
      
      df1+=df1_
      db1+=db1_
      df2+=df2_
      db2+=db2_
      dw1+=dw1_
      db3+=db3_
      dw2+=dw2_
      db4+=db4_

      df3 = df3_
      df4 = df4_
      # df5 = df5_
      # df6 = df6_
      # df7 = df7_
      # df8 = df8_
      # df9 = df9_
      # df10 = df10_
      # df11 = df11_
      # df12 = df12_
      # df13 = df13_

      cost_+= loss

    # Parameter Update  
        
    v1 = beta1*v1 + (1-beta1)*df1/batch_size # momentum update
    s1 = beta2*s1 + (1-beta2)*(df1/batch_size)**2 # RMSProp update
    f1 -= lr * v1/np.sqrt(s1+1e-7) # combine momentum and RMSProp to perform update with Adam
    
    bv1 = beta1*bv1 + (1-beta1)*db1/batch_size
    bs1 = beta2*bs1 + (1-beta2)*(db1/batch_size)**2
    b1 -= lr * bv1/np.sqrt(bs1+1e-7)
    
    v2 = beta1*v2 + (1-beta1)*df2/batch_size
    s2 = beta2*s2 + (1-beta2)*(df2/batch_size)**2
    f2 -= lr * v2/np.sqrt(s2+1e-7)
                       
    bv2 = beta1*bv2 + (1-beta1) * db2/batch_size
    bs2 = beta2*bs2 + (1-beta2)*(db2/batch_size)**2
    b2 -= lr * bv2/np.sqrt(bs2+1e-7)

    v3 = beta1*v3 + (1-beta1)*df3/batch_size
    s3 = beta2*s3 + (1-beta2)*(df3/batch_size)**2
    f3 -= lr * v3/np.sqrt(s3+1e-7)

    v4 = beta1*v4 + (1-beta1)*df4/batch_size
    s4 = beta2*s4 + (1-beta2)*(df4/batch_size)**2
    f4 -= lr * v4/np.sqrt(s4+1e-7)
    
    # The weight variables
    v14 = beta1*v14 + (1-beta1) * dw1/batch_size
    s14 = beta2*s14 + (1-beta2)*(dw1/batch_size)**2
    w1 -= lr * v14/np.sqrt(s14+1e-7)
    
    bv3 = beta1*bv3 + (1-beta1) * db3/batch_size
    bs3 = beta2*bs3 + (1-beta2)*(db3/batch_size)**2
    b3 -= lr * bv3/np.sqrt(bs3+1e-7)
    
    v15 = beta1*v15 + (1-beta1) * dw2/batch_size
    s15 = beta2*s15 + (1-beta2)*(dw2/batch_size)**2
    w2 -= lr * v15 / np.sqrt(s15+1e-7)
    
    bv4 = beta1*bv4 + (1-beta1)*db4/batch_size
    bs4 = beta2*bs4 + (1-beta2)*(db4/batch_size)**2
    b4 -= lr * bv4 / np.sqrt(bs4+1e-7)
    
    cost_ = cost_/batch_size
    cost.append(cost_)

    params = [f1, f2, f3, f4, w1, w2, b1, b2, b3, b4]
    
    return params, cost

#####################################################
##################### Training ######################
#####################################################

def train(num_classes = 10, lr = 0.01, beta1 = 0.95, beta2 = 0.99, img_dim = 28, img_depth = 1, f = 5, num_filt1 = 8, num_filt2 = 8, batch_size = 32, num_epochs = 2, save_path = 'params.pkl'):

    # Get training data
    m =50000
    X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m,1)
    X-= int(np.mean(X))
    X/= int(np.std(X))
    train_data = np.hstack((X,y_dash))

    np.random.shuffle(train_data)

    ## Initializing all the parameters
    f1, f2, f3, f4, f5, f6, f7 = (64 ,1,3,3), (64 ,64,3,3), (128, 64,3,3), (128, 128,3,3), (256, 128,3,3), (256, 256,3,3), (256, 256,3,3)
    f8, f9, f10, f11, f12, f13 = (512, 256,3,3), (512, 512,3,3), (512, 512,3,3), (512, 512,3,3), (512, 512,3,3), (512, 512,3,3)
    
    w3, w4 = (128,800), (10, 128)

    print("Initializing Filters")
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    f3 = initializeFilter(f3)
    f4 = initializeFilter(f4)
    f5 = initializeFilter(f5)
    f6 = initializeFilter(f6)
    f7 = initializeFilter(f7)
    f8 = initializeFilter(f8)
    f9 = initializeFilter(f9)
    f10 = initializeFilter(f10)
    f11 = initializeFilter(f11)
    f12 = initializeFilter(f12)
    f13 = initializeFilter(f13)

    w3 = initializeWeight(w3)
    w4 = initializeWeight(w4)

    b1 = np.zeros((f1.shape[0],1))
    b2 = np.zeros((f2.shape[0],1))
    b3 = np.zeros((w3.shape[0],1))
    b4 = np.zeros((w4.shape[0],1))

    params = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, w3, w4, b1, b2, b3, b4]

    cost = []

    print("LR:"+str(lr)+", Batch Size:"+str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x,batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))
            

    with open(save_path, 'wb') as file:
        pickle.dump(params, file)
        
    return cost

if __name__ == '__main__':
  content_img = cv2.imread("fox2.png")
  style_img = cv2.imread("starrynight.jpg")
  
  # extract 3 color channels from content and style images
  content_r = content_img[:,:,2]
  content_g = content_img[:,:,1]
  content_b = content_img[:,:,0]
  style_r = style_img[:,:,2]
  style_g = style_img[:,:,1]
  style_b = style_img[:,:,0]

  # begin convolutions
  ## Initializing all the parameters
  f1, f2 = (64 ,1,3,3), (64 ,64*1,3,3)
  w3, w4 = (128,800), (10, 128)

  print("Initializing Filters")
  f1 = initializeFilter(f1)
  f2 = initializeFilter(f2)

  w3 = initializeWeight(w3)
  w4 = initializeWeight(w4)

  b1 = np.zeros((f1.shape[0],1))
  b2 = np.zeros((f2.shape[0],1))
  b3 = np.zeros((w3.shape[0],1))
  b4 = np.zeros((w4.shape[0],1))

  params = [f1, f2, w3, w4, b1, b2, b3, b4]

  print(content_img.shape)
  print(f1.shape)

  img = np.array([content_r])


  loss = conv(np.array([content_r]), params, 1, 2, 2)
