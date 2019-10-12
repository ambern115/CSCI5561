import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import math

from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


def find_match(img1, img2):
    # To do
    
    # Use cv2 to get descriptors
    sift_img1 = cv2.xfeatures2d.SIFT_create()
    kp_img1, des_img1 = sift_img1.detectAndCompute(img1,None)

    sift_img2 = cv2.xfeatures2d.SIFT_create()
    kp_img2, des_img2 = sift_img2.detectAndCompute(img2,None)

    
    # Code used to visualize the keypoints
    #cv2.imwrite('sift_keypoints.jpg', cv2.drawKeypoints(img1,kp_img1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    #cv2.imwrite('sift_keypoints2.jpg', cv2.drawKeypoints(img2,kp_img2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

    # Find 2 nearest neighbors of each keypoint based on Euclidean distance between descriptor vectors
    nbrs_img1 = NearestNeighbors(n_neighbors=2).fit(des_img2)
    distances_img1, indices_img1 = nbrs_img1.kneighbors(des_img1)

    nbrs_img2 = NearestNeighbors(n_neighbors=2).fit(des_img1)
    distances_img2, indices_img2 = nbrs_img2.kneighbors(des_img2)

    x1 = np.empty((0,2))
    x2 = np.empty((0,2))

    # Match from left to right
    for i in range(len(distances_img1)):
        ratio = distances_img1[i][0]/distances_img1[i][1]
        if (ratio <= 0.7):
            pt1 = []
            pt2 = []

            pt1 = np.append(pt1, kp_img1[i].pt[0])
            pt1 = np.append(pt1, kp_img1[i].pt[1])
            pt2 = np.append(pt2, kp_img2[indices_img1[i][0]].pt[0])
            pt2 = np.append(pt2, kp_img2[indices_img1[i][0]].pt[1])

            x1 = np.append(x1, [pt1], axis=0)
            x2 = np.append(x2, [pt2], axis=0)

    # Match from right to left
    for i in range(len(distances_img2)):
        ratio = distances_img2[i][0]/distances_img2[i][1]
        if (ratio <= 0.7):
            pt1 = []
            pt2 = []

            pt2 = np.append(pt2, kp_img2[i].pt[0])
            pt2 = np.append(pt2, kp_img2[i].pt[1])
            pt1 = np.append(pt1, kp_img1[indices_img2[i][0]].pt[0])
            pt1 = np.append(pt1, kp_img1[indices_img2[i][0]].pt[1])

            x1 = np.append(x1, [pt1], axis=0)
            x2 = np.append(x2, [pt2], axis=0)   

    return x1, x2

# Returns the homography matrix between given point correspondences
def get_homography(x1, x2):
    A = np.empty((0,8))
    x = np.empty((0,1))
    b = np.empty((0,1))
    
    # Solve least squares between the given matrices, using homography
    for i in range(len(x2)):
        tmp_a = np.empty((0,8))
        pt1 = x1[i]
        pt2 = x2[i]
        tmp_a = np.append(tmp_a, [pt1[0], pt1[1]])
        tmp_a = np.append(tmp_a, [1, 0, 0, 0])
        tmp_a = np.append(tmp_a, [-1*pt1[0]*pt2[0]])
        tmp_a = np.append(tmp_a, [-1*pt1[1]*pt2[0]])
        A = np.append(A, [tmp_a], axis=0)

        tmp_a = np.empty((0,8))
        tmp_a = np.append(tmp_a, [0, 0, 0])
        tmp_a = np.append(tmp_a, [pt1[0], pt1[1], 1])
        tmp_a = np.append(tmp_a, [-1*pt1[0]*pt2[1]])
        tmp_a = np.append(tmp_a, [-1*pt1[1]*pt2[1]])
        A = np.append(A, [tmp_a], axis=0)

        b = np.append(b, [[pt2[0]]], axis=0)
        b = np.append(b, [[pt2[1]]], axis=0)

    #x = np.linalg.lstsq(A,b)

    try:
        tmp = np.linalg.inv(np.matmul(np.matrix.transpose(A),A))
        x = np.matmul(tmp, (np.matmul(np.matrix.transpose(A),b)))

        # Make x 3x3
        h = np.empty((0,3))
        h = np.append(h, [[x[0][0], x[1][0], x[2][0]]], axis=0)
        h = np.append(h, [[x[3][0], x[4][0], x[5][0]]], axis=0)
        h = np.append(h, [[x[6][0], x[7][0], 1]], axis=0)

        return h
    except:
        print("Error computing inverse matrix in least squares")
        return []



def visualize_homography(img1, img2, hom, x1, x2, ox1, ox2):
    # Transform all points using the transformation matrix h

    trans = np.empty((0,2))
    w = len(img1[0])
    h = len(img1)
    corners = [[0,0], [w-1, 0], [0, h-1], [w-1, h-1]]

    for i in range(4):
    # Only map 4 corners of image
        tmp = np.empty((0,1))
        tmp = np.append(tmp, [[corners[i][0]]], axis=0)
        tmp = np.append(tmp, [[corners[i][1]]], axis=0)
        tmp = np.append(tmp, [[1]], axis=0)

        tmp = np.matmul(hom, tmp)
        lmbda = 1.0/tmp[2][0]

        trans = np.append(trans, [[lmbda*tmp[0][0], lmbda*tmp[1][0]]], axis=0)

    # Draw image
    img_h=452
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    trans = trans * scale_factor1
    
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]

    ox1 = ox1 * scale_factor1
    ox2 = ox2 * scale_factor2
    ox2[:, 0] += img1_resized.shape[1]
    
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    
    for i in range(ox1.shape[0]):
        plt.plot([ox1[i, 0], ox2[i, 0]], [ox1[i, 1], ox2[i, 1]], 'y')
        plt.plot([ox1[i, 0], ox2[i, 0]], [ox1[i, 1], ox2[i, 1]], 'yo')


    for i in range(trans.shape[0]):
        plt.plot([trans[0][0], trans[1][0]], [trans[0][1], trans[1][1]], 'r')
        plt.plot([trans[1][0], trans[3][0]], [trans[1][1], trans[3][1]], 'r')
        plt.plot([trans[3][0], trans[2][0]], [trans[3][1], trans[2][1]], 'r')
        plt.plot([trans[2][0], trans[0][0]], [trans[2][1], trans[0][1]], 'r')
        
        #plt.plot([trans[0][0], trans[1][0]], [trans[0][1], trans[1][1]], 'bo')
        #plt.plot([trans[1][0], trans[2][0]], [trans[1][1], trans[2][1]], 'bo')
        #plt.plot([trans[2][0], trans[3][0]], [trans[2][1], trans[3][1]], 'bo')
        #plt.plot([trans[3][0], trans[0][0]], [trans[3][1], trans[0][1]], 'bo')
        #plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')

    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')


    
    plt.axis('off')
    plt.show()

    


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter, tmp, tar):
    # ransac_th: error threshold
    # ransac_iter: number of iterations it runs

    # Random Sampling of 4 correspondences between x1 and x2
    # Find inliers with a certain threshold
    total_points = len(x1)

    # list of indices of 4 points sampled and the number of inliers they achieved
    samples = []
    
    for iteration in range(ransac_iter):
        # Sample 4 correspondences
        inliers = 0
        rand_nums = random.sample(range(0, len(x1)), 4)

        # Add sampled numbers to a numpy matrix
        tmp_x1 = np.empty((0,2))
        tmp_x2 = np.empty((0,2))
        for n in rand_nums:
            tmp_x1 = np.append(tmp_x1, [x1[n]], axis=0)
            tmp_x2 = np.append(tmp_x2, [x2[n]], axis=0) 

        # these are used just for visualizing the 4 correspondences chosen 
        ox1 = tmp_x1
        ox2 = tmp_x2

        #visualize_find_match(tmp, tar, tmp_x1, tmp_x2)

        # Get transformation matrix for random samples
        h = get_homography(tmp_x1, tmp_x2) 
        if(len(h) != 0):
             # Save indices of this sample
            samples.append(rand_nums)

            # Count the inliers
            for i in range(len(x2)):
                # Form numpy vector from correspondance
                tmp_x = np.empty((0,1))
                tmp_x = np.append(tmp_x, [[x1[i][0]]], axis=0)
                tmp_x = np.append(tmp_x, [[x1[i][1]]], axis=0)
                tmp_x = np.append(tmp_x, [[1]], axis=0)

                # Multiply vector and transformation matrix
                tmp_vec = np.matmul(h, tmp_x)
                    
                # Normalize 
                lmbda = 1.0/tmp_vec[2][0]
                x_hat = lmbda*tmp_vec[0][0]
                y_hat = lmbda*tmp_vec[1][0]

                # Corresponding point on target image
                tar_vec = [[x2[i][0]], [x2[i][1]]]
                # Corresponding point on template image
                tmp_vec = [[x_hat], [y_hat]]

                # Computed distanced of this correspondence from the model 
                dist_from_model = np.linalg.norm(np.subtract(tar_vec, tmp_vec))

                # Determine if this correspondence is within a threshold of the model
                #in_x1 = np.empty((0,2))
                #in_x2 = np.empty((0,2))
                if ((dist_from_model) <= ransac_thr):
                    inliers = inliers + 1
                    #tmp_in_x1 = np.append(tmp_in_x1, [x1[i]], axis=0)
                    #tmp_in_x2 = np.append(tmp_in_x2, [x2[i]], axis=0)
            # Add number of inliers to the samples taken  
            samples[len(samples)-1].append(inliers)
            samples[len(samples)-1].append(h)


    # Determine which model fit best based on max number of inliers
    max_inliers = 0
    max_idx = -1 # index of the best model in the samples array
    for i in range(len(samples)):
        if (samples[i][4] >= max_inliers):
            max_inliers = samples[i][4]
            max_idx = i 
    print(max_inliers)
    
    # Retrieve this model again
    in_x1 = np.empty((0,2)) 
    in_x2 = np.empty((0,2))
    tmp_x1 = np.empty((0,2))
    tmp_x2 = np.empty((0,2))
    for i in range(4):
        tmp_x1 = np.append(in_x1, [x1[samples[max_idx][i]]], axis=0) 
        tmp_x2 = np.append(in_x2, [x2[samples[max_idx][i]]], axis=0)

    ox1 = tmp_x1
    ox2 = tmp_x2

    # Get transformation matrix for model again
    h = samples[max_idx][5]

    # Repeat was was done in the loop before   
    for i in range(len(x2)):
        # Form numpy vector from correspondance
        tmp_x = np.empty((0,1))
        tmp_x = np.append(tmp_x, [[x1[i][0]]], axis=0)
        tmp_x = np.append(tmp_x, [[x1[i][1]]], axis=0)
        tmp_x = np.append(tmp_x, [[1]], axis=0)

        # Multiply vector and transformation matrix
        tmp_vec = np.matmul(h, tmp_x)
                
        # Normalize 
        lmbda = 1.0/tmp_vec[2][0]
        x_hat = lmbda*tmp_vec[0][0]
        y_hat = lmbda*tmp_vec[1][0]

        # Corresponding point on target image
        tar_vec = [[x2[i][0]], [x2[i][1]]]
        # Corresponding point on template image
        tmp_vec = [[x_hat], [y_hat]]

        # Computed distanced of this correspondence from the model 
        dist_from_model = np.linalg.norm(np.subtract(tar_vec, tmp_vec))

        # Determine if this correspondence is within a threshold of the model
        #in_x1 = np.empty((0,2))
        #in_x2 = np.empty((0,2))
        if ((dist_from_model) <= ransac_thr):
            in_x1 = np.append(in_x1, [x1[i]], axis=0)
            in_x2 = np.append(in_x2, [np.array([x_hat, y_hat])], axis=0)

    visualize_homography(tmp, tar, h, in_x1, in_x2, ox1, ox2)  

    return h  
    

def warp_image(img, A, output_size):
    # To do
    return img_warped


def align_image(template, target, A):
    # To do
    return A_refined


def track_multi_frames(template, img_list):
    # To do
    return A_list



def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    #template = cv2.imread('./einstein.jpg', 0)
    ##template = cv2.imread('./dalton1.jpg', 0)
    #dalton = cv2.imread('./dalton2.jpg', 0)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    A = align_image_using_feature(x1, x2, 6, 50, template, target_list[1])

    img_warped = warp_image(target_list[0], A, template.shape)
    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    # A_refined, errors = align_image(template, target_list[0], A)
    # visualize_align_image(template, target_list[0], A, A_refined, errors)

    # A_list = track_multi_frames(template, target_list)
    # visualize_track_multi_frames(template, target_list, A_list)


