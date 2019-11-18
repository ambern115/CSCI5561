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


def compute_dsift(img, stride, size):
    # To do
    # problem: stride is amount of pixels to move right and down on image at each computation
    # size is still size of each 'block'

    # get number of descriptors in img (w and h)
    descs_h = math.floor(math.floor(len(img)/size) * (size/stride))
    descs_w = math.floor(math.floor(len(img[0])/size) * (size/stride))

    # make sure h and w don't go over the edge of the image...
    if ((descs_h-1)*stride+size >= len(img)):
        descs_h = descs_h-1
    if ((descs_w-1)*stride+size >= len(img[0])):
        descs_w = descs_w-1

    dense_feature = np.empty((0, 128))

    half = size/2.0
    extractor = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(half, half, _size=(size/1.0))]

    # plt.imshow(img)
    # plt.show()
    for i in range(descs_h): # what row we're in
        for j in range(descs_w): # what col we're in
            box_img = np.empty((0, size))
            # Fill in box img with pixel from the current square
            for u in range(size):
                img_row = np.empty((0))
                went_oob = False
                for v in range(size):
                    try:
                        px = img[i*stride+u][j*stride+v]
                        
                        img_row = np.append(img_row, [px[0]])
                       
                        #box_img[u][v] = img[i*stride+u][j*stride+v][0]
                    except:
                        went_oob = True
                       
                        
                        # probably run out of img
                if (went_oob == False):
                    box_img = np.append(box_img, [img_row], axis=0)


            box_img = box_img.astype('uint8')


            # Retrieve SIFT descriptor for this local patch
            kps, des = extractor.compute(box_img, kp)

            dense_feature = np.append(dense_feature, [des[0]], axis=0)
            
            # print(kp)
            # print(des)

    return dense_feature


def get_tiny_image(img, output_size):
    w = output_size[0]
    h = output_size[1]

    #print(img)
    img = cv2.resize(img, (w , h));


    vectorized = np.empty((0,1))
    avg_val = 0

    
    for i in range(h):
        for j in range(w):
            vectorized = np.append(vectorized, img[i][j][0])
            avg_val += img[i][j][0]

    

    # vectorized = np.reshape(vectorized, ((h*w), 1))
    
    vectorized -= np.mean(vectorized).astype("uint8")
    
    length = np.linalg.norm(vectorized)

    if (length != 0):
        vectorized /= length
    

    return vectorized


def predict_knn(feature_train, label_train, feature_test, k):
    # For each example of data

    label_test_pred = []
    len_training = len(feature_train)
    # for each item we need to find a label for...
    for i in range(len(feature_test)): 
        test_f = feature_test[i]
        distances = []
        first_item = True
        for j in range(len_training): # for each training item
            train_f = feature_train[j]

            # Calculate distance between the test img vector and the training img vector
            #dist = np.linalg.norm(test_f-train_f)
            tmp = train_f-test_f
            dist = np.linalg.norm(tmp)
            
            
            # Add dist and idx to list, sorting by distance
            if (first_item == False): 
                spot_found = False
                for idx in range(len(distances)): # for each measured item
                    if (dist <= distances[idx][0]): # if current distance is <= the next one in line, add it to the array here
                        distances.insert(idx, [dist, j])
                        spot_found = True
                        break
                if (spot_found == False): # If this item's distance is larger than all others so far
                    distances.append([dist, j])
            else:
                distances.append([dist, j])
                first_item = False

        # Get the most frequently appearing label (mode) from k nearest neighbors
        # Count the ocurrances of unique labels
        label_names = []
        label_cnt = []
        # potential problem with k being too large here? had error at 20
        # distances length is not consistent even though it should be 

        for j in range(k):
            # See if this label is already being counted 
            try:
                idx = label_names.index(label_train[distances[j][1]])
                label_cnt[idx] += 1
            except:
                # Add new nn label to list of label names
                label_names.append(label_train[distances[j][1]])
                label_cnt.append(1)

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

    # get images from img_train_list and img_test_list
    train_list = []
    test_list = []

    # Convert images in lists into tiny images
    for img in img_train_list:
        train_list.append(get_tiny_image(cv2.imread(img), [16,16]))
    for img in img_test_list:
        test_list.append(get_tiny_image(cv2.imread(img), [16,16]))


    # Predict labels for test images
    predicted_labels = predict_knn(train_list, label_train_list, test_list, 5)

    inc = 0

    confusion = np.zeros((len(label_classes), len(label_classes)))

    # Create confusion matrix
    correct_predictions = 0.0
    for i in range(len(label_classes)):
        row_label = label_classes[i]
        occurances = 0.0
        # loop through img_test_list to get total # of occurances of this image
        # this is the row's label
        for label in label_test_list:
            if (label == row_label): 
                occurances += 1.0
        total = 0.0
        
        for j in range(len(label_classes)):
            # these are the labels in the columns
            # determine number of occurances of label j in the predictions array for the label i
            col_label = label_classes[j]
            predicted_occr = 0.0 # number of times it predicted this col label for the row's label
            for l in range(len(label_test_list)):
                if (label_test_list[l] == row_label): # if correct row label
                    # add to predicted if the corresponding prediction = col's label
                    if (predicted_labels[l] == col_label):
                        predicted_occr += 1.0
                        total += 1.0
                
            color = predicted_occr/occurances

            print(inc)
            inc += 1
            
            # If row and col labels match
            if (row_label == col_label):
                correct_predictions += color

                # color the corresponding pixel
                #  for p_row in range(len(label_classes)):
                #    for p_col in range(len(label_classes)):
            confusion[i][j] = color
                        
        print(total)



            # loop through labels for im in test list, counting # of occurances of i

    accuracy = correct_predictions/(len(label_classes))
            
    


    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    print("3===D", dic_size)
    print("<(^^)>", dense_feature_list.shape)
    vocab = KMeans(n_clusters=dic_size, n_init=10, max_iter=300).fit(dense_feature_list)
    print("AH SHIT")
    # fit_predict(self, X[, y, sample_weight]) 	Compute cluster centers and predict cluster index for each sample.
    # predict(self, X[, sample_weight]) 	Predict the closest cluster each sample in X belongs to.
    
    #np.savetxt("clusters_50.txt", vocab.cluster_centers_)
    
    return vocab.cluster_centers_


def compute_bow(feature, vocab):
    # To do
    clusters_chosen = []

    neghbrs = NearestNeighbors(n_neighbors=1).fit(vocab) 
    distance, index = neghbrs.kneighbors(feature)

    for feat in feature:
        # find cluster nearest to this feature
        neghbrs = NearestNeighbors(n_neighbors=1).fit(vocab)
        
        distance, index = neghbrs.kneighbors([feat])
        

        clusters_chosen.append(index[0][0])

    # Determine the mode of clusters chosen
    temp = []
    for i in range(len(vocab)):
        temp.append(0)

    for i in clusters_chosen:
        temp[i] += 1
    
    bow_feature = np.array(temp)
    
    # normalize feature
    # length = np.linalg.norm(bow_feature)
    # print(length)
    # print(bow_feature)

    # if (length > 0):
    #     for i in range(len(bow_feature)):
    #         bow_feature[i] /= length
    # print(bow_feature)

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    
    # get images from img_train_list and img_test_list
    train_list = []
    test_list = []

    # Convert images in lists into tiny images
    for img in img_train_list:
        train_list.append(cv2.imread(img))
    for img in img_test_list:
        test_list.append(cv2.imread(img))


    # I precomputed train and test bows
    bows_trained_bad = np.loadtxt("trained_bows.txt")
    bows_test_bad = np.loadtxt("test_bows.txt")

    # forgot to normalize them
    # normalize them all
    bows_trained = np.empty((0,50))
    bows_test = np.empty((0,50))
    for vec in bows_trained_bad:
        length = np.linalg.norm(vec)
        bows_trained = np.append(bows_trained, [(vec / length)], axis=0)
    for vec in bows_test_bad:
        length = np.linalg.norm(vec)
        bows_test = np.append(bows_test, [(vec / length)], axis=0)

    

    # Predict labels for test images
    predicted_labels = predict_knn(bows_trained, label_train_list, bows_test, 20)


    confusion = np.zeros((len(label_classes), len(label_classes)))

    inc = 0
    # Create confusion matrix
    correct_predictions = 0.0
    for i in range(len(label_classes)):
        row_label = label_classes[i]
        occurances = 0.0
        # loop through img_test_list to get total # of occurances of this image
        # this is the row's label
        for label in label_test_list:
            if (label == row_label): 
                occurances += 1.0
        total = 0.0
        
        for j in range(len(label_classes)):
            # these are the labels in the columns
            # determine number of occurances of label j in the predictions array for the label i
            col_label = label_classes[j]
            predicted_occr = 0.0 # number of times it predicted this col label for the row's label
            for l in range(len(label_test_list)):
                if (label_test_list[l] == row_label): # if correct row label
                    # add to predicted if the corresponding prediction = col's label
                    if (predicted_labels[l] == col_label):
                        predicted_occr += 1.0
                        total += 1.0
                
            color = predicted_occr/occurances
            
            # If row and col labels match
            if (row_label == col_label):
                correct_predictions += color

                # color the corresponding pixel
                #  for p_row in range(len(label_classes)):
                #    for p_col in range(len(label_classes)):
            confusion[i][j] = color
                        
        



            # loop through labels for im in test list, counting # of occurances of i

    accuracy = correct_predictions/(len(label_classes))
            

    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test):
    # To do
    classifier = LinearSVC(random_state=0)
    classifier.fit(feature_train, label_train)

    label_test_pred = classifier.predict(feature_test)

    # label_test_pred = np.empty((0, len(label_train)))
    # for feat in feature_test:
    #     label_test_pred = np.append(label_test_pred, [classifier.predict(feat)], axis=0)

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
        # get images from img_train_list and img_test_list
    train_list = []
    test_list = []

    # Convert images in lists into tiny images
    for img in img_train_list:
        train_list.append(cv2.imread(img))
    for img in img_test_list:
        test_list.append(cv2.imread(img))

    # I precomputed train and test bows
    bows_trained = np.loadtxt("trained_bows.txt")
    bows_test = np.loadtxt("test_bows.txt")

    # bows_trained = np.empty((0,50))
    # bows_test = np.empty((0,50))
    # for vec in bows_trained_bad:
    #     length = np.linalg.norm(vec)
    #     bows_trained = np.append(bows_trained, [(vec / length)], axis=0)
    # print(bows_trained)
    # for vec in bows_test_bad:
    #     length = np.linalg.norm(vec)
    #     bows_test = np.append(bows_test, [(vec / length)], axis=0)
    # print(bows_test)


    # Predict labels for test images
    predicted_labels = predict_svm(bows_trained, label_train_list, bows_test)

    inc = 0
    confusion = np.zeros((len(label_classes), len(label_classes)))

    # Create confusion matrix
    correct_predictions = 0.0
    for i in range(len(label_classes)):
        row_label = label_classes[i]
        occurances = 0.0
        # loop through img_test_list to get total # of occurances of this image
        # this is the row's label
        for label in label_test_list:
            if (label == row_label): 
                occurances += 1.0
        total = 0.0
        
        for j in range(len(label_classes)):
            # these are the labels in the columns
            # determine number of occurances of label j in the predictions array for the label i
            col_label = label_classes[j]
            predicted_occr = 0.0 # number of times it predicted this col label for the row's label
            for l in range(len(label_test_list)):
                if (label_test_list[l] == row_label): # if correct row label
                    # add to predicted if the corresponding prediction = col's label
                    if (predicted_labels[l] == col_label):
                        predicted_occr += 1.0
                        total += 1.0
                
            color = predicted_occr/occurances
            
            
            # If row and col labels match
            if (row_label == col_label):
                correct_predictions += color

                # color the corresponding pixel
                #  for p_row in range(len(label_classes)):
                #    for p_col in range(len(label_classes)):
            confusion[i][j] = color
                        
        



            # loop through labels for im in test list, counting # of occurances of i

    accuracy = correct_predictions/(len(label_classes))    


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


def save_descriptors(descriptors, file_name):
    file_string = ""
    for desc in descriptors:
        desc = np.nan_to_num(desc)
        line = ""
        for num in desc:
            line += str(num) + " "
        file_string += line[:-1] + "\n"
    file_string = file_string[:-1]

    file = open(file_name,"w") 
    file.write(file_string)
    file.close()
    #np.savetxt(file_name, [file_string], fmt="%s")

def extract_descriptors(file_name):
    file = open(file_name, "r")


    # extract descriptors for images
    line = "x"
    img_desc = np.empty((0,128))
    line = file.readline()
    while (line != ''):
        desc = np.empty((0,0))

        # parse floats in this descriptor
        float_list = line.split(' ')
        
        for flt in float_list:
            if (flt != '\n'):
                desc = np.append(desc, flt)

        desc = desc.astype(np.float)
        #desc = np.nan_to_num(desc)
        img_desc = np.append(img_desc, [desc], axis=0)
        line = file.readline()


    print(img_desc.shape)

    return img_desc
        
# def no_nans(dsift_desc):
#     for arr in dsift_desc:


if __name__ == '__main__':


    


    # # fp = '/home/amber/Desktop/CSCI5561/Hwk3/'
    # img1 = cv2.imread('image_0029.jpg')
    # compute_dsift(img1, 20, 40)

    # img2 = cv2.imread(fp + 'image_0028.jpg')
    # img3 = cv2.imread(fp + 'image_0009.jpg')
    # img4 = cv2.imread(fp + 'image_0005.jpg')
    
    #sfs = []
    

    # sze = 50
    # sfs = np.empty((0,128))
    # sfs = np.append(sfs, compute_dsift(img2, sze, sze).astype(np.float), axis=0)
    # sfs = np.append(sfs, compute_dsift(img1, sze, sze).astype(np.float), axis=0)
    # sfs = np.append(sfs, compute_dsift(img3, sze, sze).astype(np.float), axis=0)
    # sfs = np.append(sfs, compute_dsift(img4, sze, sze).astype(np.float), axis=0)
    
    # sfs.append([compute_dsift(img2, sze, sze)])
    # sfs.append([compute_dsift(img1, sze, sze)])
    # sfs.append([compute_dsift(cv2.imread('./image_0009.jpg'), sze, sze)])
    # sfs.append([compute_dsift(cv2.imread('./image_0005.jpg'), sze, sze)])

    #sfs = np.array(sfs)

    # print(sfs)
    # print(sfs.shape)

    #save_descriptors(sfs, "test.txt")

    # ex = extract_descriptors("test.txt")
   
    # # # sfs = extract_descriptors("test.txt")
    
    # # # print(len(sfs))
    # # # print(sfs)
   
    # dic = build_visual_dictionary(sfs, 5)

    # clusters = np.loadtxt("clusters.txt")
    # print(clusters)
    # np.savetxt("cluster_test.txt", clusters)
    # print(dic)
    # print(dic.labels_)
    # print(dic.cluster_centers_)

 
    # save_descriptors(sfs, "dsift_test.txt")
    # print("SAVED")

    # print(extracted)
    # print(np.isnan(extracted).any())


    # new_descss = extract_descriptors("dsift.txt")
    

    # save_descriptors(new_descss, "dsift2.txt")




    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("/home/amber/Desktop/CSCI5561/Hwk3//scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    # # # # Get images from img_train_list and img_test_list
    # train_list = []
    # test_list = []

    # for img in img_train_list:
    #     train_list.append(cv2.imread(img))
    # for img in img_test_list:
    #     test_list.append(cv2.imread(img))
    
    # # dsift_imgs = np.empty((0,128))
    # # # print("computing dsift_imgs")
    # # for img in test_list:
    # #     dsift_imgs = np.append(dsift_imgs, compute_dsift(img, 20, 40), axis=0)

    # vocab = np.loadtxt("/home/amber/Desktop/CSCI5561/Hwk3/clusters_50.txt")

    # # compute BOWs for each training image
    # test_bows = np.empty((0,50))
    # inc = 0
    # for img in test_list:
    #     test_bows = np.append(test_bows, [compute_bow(compute_dsift(img, 20, 40), vocab)], axis=0)
    #     inc += 1
    #     print(inc)
        
    # print("SAVING")
       
    # np.savetxt("test_bows.txt", test_bows)
    # print("saved")

    # save_descriptors(dsift_imgs, "dsift.txt")
    # print("SAVED")

    # dsift_imgs = extract_descriptors("dsift.txt")

    # print("getting kmeans")
    # dic = build_visual_dictionary(dsift_imgs, 50)

    # print(dic)
    # print(dic.cluster_centers_)
    # print(dic.labels_)

    # vocab = np.loadtxt("/home/amber/Desktop/CSCI5561/Hwk3/clusters_50.txt")
    # print(vocab.shape)
    # sift1 = compute_dsift(train_list[0], 20, 40)
    # sift2 = compute_dsift(train_list[10], 20, 40)
    # sift3 = compute_dsift(train_list[20], 20, 40)
    # # print(sift.shape)

    # # for 
    # bows = np.empty((0, 50))
    # bows = np.append(bows, [compute_bow(sift1, vocab)], axis=0)
    # bows = np.append(bows, [compute_bow(sift2, vocab)], axis=0)
    # bows = np.append(bows, [compute_bow(sift3, vocab)], axis=0)


    # print("saving")
    # # write dsift_imgs to a file (so it doesn't have to be computed every time...)
    # save_descriptors(dsift_imgs, "dsift.txt")
    # print("SAVED")

    # extracted = extract_descriptors("dsift.txt")

    # print("extracted")

    # dic = build_visual_dictionary(extracted, 12)

    # print(dic)
    # print(dic.labels_)
    # print(dic.cluster_centers_)
    #save_descriptors(extracted, "dsift2.txt")

    # loaded_dsift = np.loadtxt("dsift.txt", 'float')

    # #print(predict_knn(training_imgs, label_train, testing_imgs, 5))

    # # To do: replace with your dataset path
    # #label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    

    # classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    # classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    # classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)



