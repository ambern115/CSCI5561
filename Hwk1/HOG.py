import cv2
import math 
import numpy as np
import matplotlib.pyplot as plt

def get_differential_filter():
    filter_x = [[1.0,0.0,-1.0],[1.0,0.0,-1.0],[1.0,0.0,-1.0]]
    filter_y = [[1.0,1.0,1.0],[0.0,0.0,0.0],[-1.0,-1.0,-1.0]]

    return filter_x, filter_y


def filter_image(im, filtr):
    print("Filtering Image")
    im = [[1,2,4,5],[6,1,0,1],[0,0,1,1],[8,0,2,4]]
    if (len(im) == 0 or len(im[0]) == 0):
        print("Error: image to be filtered is too small.")
        return im
    if (len(filtr) == 0 or len(filtr[0]) == 0):
        print("Error: filter size too small.")
        return im

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
            row_filtered.append(temp_sum)
        im_filtered.append(row_filtered)

    #im_filtered = scipy.signal.convolve2d(im, filtr)

    return im_filtered


def get_gradient(im_dx, im_dy):
    print("Retrieving Gradients")

    # Calculate the magnitude gradient
    x_squared = np.square(im_dx)
    y_squared = np.square(im_dy)
    grad_mag = np.sqrt(np.add(x_squared,y_squared))

    # Caluculate the angle gradient
    grad_angle = np.arctan2(im_dy, im_dx)

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    print("Building Histogram")
    if (len(grad_mag) == 0 or len(grad_angle) == 0):
        print("Error: gradient(s) too small")
        empty = [[[]]]
        return empty

    im_h = len(grad_mag)
    im_w = len(grad_mag[0])

    # Get number of cells along x and y axes
    M = math.floor(im_h/cell_size)
    N = math.floor(im_w/cell_size)

    ori_histo = []

    for i in range(M):
        ori_row = []
        for j in range(N):
            ori_cell = []
            for bn in range(cell_size):
                bin_sum = 0
                for u in range(cell_size):
                    for v in range(cell_size):
                        # Check if angle within bin's angle range
                        angle = grad_angle[u+i*cell_size][v+j*cell_size]
                        print(angle)
                        # make sure max is 180...
                        if(bn == 0):
                            if(angle >= 165 or angle < 15):
                                bin_sum += grad_mag[u+i*cell_size][v+j*cell_size]
                        elif(bn == 1):
                            if(angle < 45):
                                bin_sum += grad_mag[u+i*cell_size][v+j*cell_size]
                        elif(bn == 2):
                            if(angle < 75):
                                bin_sum += grad_mag[u+i*cell_size][v+j*cell_size]
                        elif(bn == 3):
                            if(angle < 105):
                                bin_sum += grad_mag[u+i*cell_size][v+j*cell_size]
                        elif(bn == 4):
                            if(angle < 135):
                                bin_sum += grad_mag[u+i*cell_size][v+j*cell_size]
                        elif(bn == 5):
                            if(angle < 165):
                                bin_sum += grad_mag[u+i*cell_size][v+j*cell_size]
                ori_cell.append(bin_sum)
            ori_row.append(ori_cell)
        ori_histo.append(ori_row)

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do

    # get differential images
    diff_filter_x, diff_filter_y = get_differential_filter();

    diff_x = filter_image(im, diff_filter_x)
    diff_y = filter_image(im, diff_filter_y)

    # get gradients 
    grad_mag, grad_angle = get_gradient(diff_x, diff_y)

    histogram = build_histogram(grad_mag, grad_angle, 8)

    #visualize(grad_angle)

    # visualize to verify
    #visualize_hog(im, hog, 8, 2)
    #visualize_hog(im, im, 8, 2)

    #return hog
    return im

def visualize(im):
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.show()

# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    im = cv2.imread('dog.jpeg', 0)
    hog = extract_hog(im)


