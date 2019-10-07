import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
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

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    return A

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
        print("plot made" + str(i) + " " + str(x1.shape[0]))
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
    template = cv2.imread('./dalton1.jpg', 0)
    dalton = cv2.imread('./dalton2.jpg', 0)
    

    x1, x2 = find_match(template, dalton)
    visualize_find_match(template, dalton, x1, x2)


    # A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    # img_warped = warp_image(target_list[0], A, template.shape)
    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()

    # A_refined, errors = align_image(template, target_list[0], A)
    # visualize_align_image(template, target_list[0], A, A_refined, errors)

    # A_list = track_multi_frames(template, target_list)
    # visualize_track_multi_frames(template, target_list, A_list)


