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



def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img):
    # To do
    return dense_feature


def get_tiny_image(img, output_size):
    w = output_size[0]
    h = output_size[1]

    cv2.resize(img, (w , h));

    vectorized = np.empty((0,0))
    avg_val = 0

    
    for i in range(h):
        for j in range(w):
            vectorized = np.append(vectorized, img[h][w][0])
            avg_val += img[h][w][0]

    vectorized = np.reshape(vectorized, ((h*w), 1))
    
    vectorized -= avg_val
    
    length = np.linalg.norm(vectorized)

    vectorized /= np.linalg.norm(vectorized)

    return vectorized


def predict_knn(feature_train, label_train, feature_test, k):
    # For each example of data

    label_test_pred = []

    for i in range(len(feature_test)):
        test_f = feature_train[i]
        distances = []
        for j in range(len(feature_train)):
            if (i != j):
                train_f = feature_train[j]

                # Calculate distance between the query example and the current example from the data
                dist = np.linalg.norm(test_f-train_f)
                
                # Make sure the list is sorted by distance
                if (len(distances) > 0):
                    for idx in range(len(distances)):
                        if (dist <= distances[idx][0]): # Insert
                            distances.insert(idx, [dist, j])
                            break
                else:
                    distances.append([dist, j])

        # Get the most frequently appearing label from k nearest neighbors

        # Count the ocurrances of unique labels
        label_names = []
        label_cnt = []
        for j in range(k):
            # See if this label is already being counted 
            try:
                idx = label_names.index(label_train[distances[j][1]])
                label_cnt[idx] += 1
            except:
                label_names.append(label_train[distances[j][1]])
                label_cnt.append(1)

        print(label_names)
        print(label_cnt)
        
        # Retrieve label with max occurance
        max_idx = 0
        max_cnt = 0
        for j in range(len(label_cnt)):
            if (label_cnt[j] > max_cnt):
                max_cnt = label_cnt[j]
                max_idx = j
        final_label = label_names[max_idx]

        label_test_pred.append(final_label)

    # To do
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    return vocab


def compute_bow(feature, vocab):
    # To do
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('./image_0029.jpg')
    # img = img.astype('float') / 255.0
    training_imgs = []
    testing_imgs = []

    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Bedroom/image_0003.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Bedroom/image_0004.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Bedroom/image_0011.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Bedroom/image_0006.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Bedroom/image_0007.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Bedroom/image_0008.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Coast/image_0001.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Coast/image_0003.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Coast/image_0004.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Coast/image_0005.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Coast/image_0011.jpg'), [16,16]))
    training_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/test/Coast/image_0007.jpg'), [16,16]))

    label_train = ["Bedroom","Bedroom","Bedroom","Bedroom","Bedroom","Bedroom","Coast","Coast","Coast","Coast","Coast","Coast"]

    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Bedroom/image_0001.jpg'), [16,16]))
    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Bedroom/image_0002.jpg'), [16,16]))
    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Bedroom/image_0005.jpg'), [16,16]))
    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Bedroom/image_0009.jpg'), [16,16]))
    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Bedroom/image_0010.jpg'), [16,16]))
    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Coast/image_0006.jpg'), [16,16]))
    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Coast/image_0010.jpg'), [16,16]))
    testing_imgs.append(get_tiny_image(cv2.imread('scene_classification_data/train/Coast/image_0015.jpg'), [16,16]))
    

    print(predict_knn(training_imgs, label_train, testing_imgs, 5))
    


    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




