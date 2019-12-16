import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
from scipy.linalg import svd
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
import random # Used in RANSAC

def find_match(img1, img2):
  # Use cv2 to get descriptors
  sift_img1 = cv2.xfeatures2d.SIFT_create()
  kp_img1, des_img1 = sift_img1.detectAndCompute(img1,None)

  sift_img2 = cv2.xfeatures2d.SIFT_create()
  kp_img2, des_img2 = sift_img2.detectAndCompute(img2,None)
    
  # # Code used to visualize the keypoints
  # cv2.imwrite('sift_keypoints.jpg', cv2.drawKeypoints(img1,kp_img1,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
  # cv2.imwrite('sift_keypoints2.jpg', cv2.drawKeypoints(img2,kp_img2,img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))

  # Find 2 nearest neighbors of each keypoint based on Euclidean distance between descriptor vectors
  nbrs_img1 = NearestNeighbors(n_neighbors=2).fit(des_img2)
  distances_img1, indices_img1 = nbrs_img1.kneighbors(des_img1)

  nbrs_img2 = NearestNeighbors(n_neighbors=2).fit(des_img1)
  distances_img2, indices_img2 = nbrs_img2.kneighbors(des_img2)

  x1_l = np.empty((0,2)) # left to right
  x2_l = np.empty((0,2)) # left to right
  x1_r = np.empty((0,2)) # right to left
  x2_r = np.empty((0,2)) # right to left
  # Match from left to right
  for i in range(len(distances_img1)):
    # Filter by ratio test
    ratio = distances_img1[i][0]/distances_img1[i][1]
    if (ratio < 0.7):
      # If points unique enough, add them
      pt1 = []
      pt2 = []

      pt1 = np.append(pt1, kp_img1[i].pt[0])
      pt1 = np.append(pt1, kp_img1[i].pt[1])
      pt2 = np.append(pt2, kp_img2[indices_img1[i][0]].pt[0])
      pt2 = np.append(pt2, kp_img2[indices_img1[i][0]].pt[1])

      x1_l = np.append(x1_l, [pt1], axis=0)
      x2_l = np.append(x2_l, [pt2], axis=0)

  # Match from right to left
  for i in range(len(distances_img2)):
    # Filter by ratio test
    ratio = distances_img2[i][0]/distances_img2[i][1]
    if (ratio < 0.7):
      pt1 = []
      pt2 = []

      pt2 = np.append(pt2, kp_img2[i].pt[0])
      pt2 = np.append(pt2, kp_img2[i].pt[1])
      pt1 = np.append(pt1, kp_img1[indices_img2[i][0]].pt[0])
      pt1 = np.append(pt1, kp_img1[indices_img2[i][0]].pt[1])

      x1_r = np.append(x1_r, [pt1], axis=0)
      x2_r = np.append(x2_r, [pt2], axis=0)

  pts1 = np.empty((0,2))
  pts2 = np.empty((0,2))

  # Bi directional consistency check
  # Filter out points that don't match in both directions
  for i in range(len(x1_l)):
    point1 = x1_l[i]
    for j in range(len(x1_r)):
      point2 = x1_r[j]
      if (point1[0] == point2[0] and point1[1] == point2[1]):
        # Check that corresponding points in x2 match
        if (x2_l[i][0] == x2_r[j][0] and x2_l[i][1] == x2_r[j][1]):
          pts1 = np.append(pts1, [x1_l[i]], axis=0)
          pts2 = np.append(pts2, [x2_l[i]], axis=0)

  return pts1, pts2


def get_homography(x1, x2):
  A = np.empty((0,9))

  # Solve least squares between the given matrices
  for i in range(len(x2)):
    tmp_a = np.empty((0,9))
    pt1 = x1[i]
    pt2 = x2[i]
    tmp_a = np.append(tmp_a, [pt1[0]*pt2[0], pt1[1]*pt2[0], pt2[0], pt1[0]*pt2[1], pt1[1]*pt2[1], pt2[1], pt1[0], pt1[1], 1.0])
    A = np.append(A, [tmp_a], axis=0)

  try:
    ns = null_space(A)

    # Make x 3x3
    h = np.empty((0,3))
    h = np.append(h, [[ns[0][0], ns[1][0], ns[2][0]]], axis=0)
    h = np.append(h, [[ns[3][0], ns[4][0], ns[5][0]]], axis=0)
    h = np.append(h, [[ns[6][0], ns[7][0], ns[8][0]]], axis=0)

    return h
  except:
    print("Error computing homography")
    return []


def compute_F(pts1, pts2):
  ransac_itr = 150
  ransac_thr = 0.05

  # Use RANSAC 8 point algorithm to estimate F matrix from points
 
  # Random Sampling of 8 correspondences between pts1 and pts2
  # Find inliers with a certain threshold

  total_points = len(pts1)

  zero_m = np.zeros((3,3))

  # list of indices of 8 points sampled and the number of inliers they achieved
  samples = []
  
  for iteration in range(ransac_itr):
    # Sample 8 correspondences
    inliers = 0
    rand_nums = random.sample(range(0, total_points), 8)

    # Add sampled numbers to a numpy matrix
    tmp_pts1 = np.empty((0,2))
    tmp_pts2 = np.empty((0,2))
    for n in rand_nums:
      tmp_pts1 = np.append(tmp_pts1, [pts1[n]], axis=0)
      tmp_pts2 = np.append(tmp_pts2, [pts2[n]], axis=0) 

    # these are used just for visualizing the 4 correspondences chosen 
    ox1 = tmp_pts1
    ox2 = tmp_pts2

    #visualize_find_match(tmp, tar, tmp_x1, tmp_x2)

    # Get transformation matrix for random samples
    h = get_homography(tmp_pts1, tmp_pts2) 
    if(len(h) != 0):
      # Save indices of this sample
      samples.append(rand_nums)

      # Count the inliers
      for i in range(len(pts2)):
        # Form numpy vector from correspondance
        pt1 = pts1[i]
        pt2 = pts2[i]

        tmp_x = np.zeros((0,3))

        tmp_x = np.append(tmp_x, [[pt1[0]*pt2[0], pt1[1]*pt2[0], pt2[0]]], axis=0)
        tmp_x = np.append(tmp_x, [[pt1[0]*pt2[1], pt1[1]*pt2[1], pt2[1]]], axis=0)
        tmp_x = np.append(tmp_x, [[pt1[0], pt1[1], 1.0]], axis=0)

        u = np.zeros((3,1))
        u[0][0] = pt1[0]
        u[1][0] = pt1[1]
        u[2][0] = 1.0

        # Multiply vectors and transformation matrix
        tmp_vec = np.dot(np.array([pt2[0],pt2[1],1.0]), h)
        tmp_vec = np.dot(tmp_vec, u)
 
        # # Normalize
        # lmbda = 1.0/tmp_vec[2][0]
        # x_hat = lmbda*tmp_vec[0][0]
        # y_hat = lmbda*tmp_vec[1][0]

        # Computed distance of this correspondence from the model 
        dist_from_model = np.linalg.norm(tmp_vec)

        # Determine if this correspondence is within a threshold of the model
        #in_x1 = np.empty((0,2))
        #in_x2 = np.empty((0,2))
        if ((dist_from_model) <= ransac_thr):
          inliers = inliers + 1
      # Add number of inliers to the samples taken  
      samples[len(samples)-1].append(inliers)
      samples[len(samples)-1].append(h)


  # Determine which model fit best based on max number of inliers
  max_inliers = 0
  max_idx = -1 # index of the best model in the samples array
  for i in range(len(samples)):
    if (samples[i][8] >= max_inliers):
      max_inliers = samples[i][8]
      max_idx = i 
  print(max_inliers)
  
  # Retrieve this model again
  in_x1 = np.empty((0,2)) 
  in_x2 = np.empty((0,2))
  tmp_x1 = np.empty((0,2))
  tmp_x2 = np.empty((0,2))
  for i in range(8):
    tmp_x1 = np.append(in_x1, [pts1[samples[max_idx][i]]], axis=0) 
    tmp_x2 = np.append(in_x2, [pts2[samples[max_idx][i]]], axis=0)

  # Get transformation matrix for model again
  F = samples[max_idx][9]

  # make rank 2
  u, s, v = svd(F)
  new_s = np.zeros((3,3))
  # get smallest eigenvalue of the diagonal and set to 0
  if (s[0] < s[1] and s[0] < s[2]):
    new_s[1][1] = s[1]
    new_s[2][2] = s[2]
  elif(s[1] < s[2]):
    new_s[0][0] = s[0]
    new_s[2][2] = s[2]
  else:
    new_s[0][0] = s[0]
    new_s[1][1] = s[1]
  
  F = np.dot(np.dot(u, new_s), v)
  
  print(np.linalg.matrix_rank(F))

  return F

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)

    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    nnz = 0
    #tmp = vh[nnz:] 
    tmp = vh[3]
    ns = tmp.T

    g = np.dot(A,ns)
    #ns = tmp.conj().T

    return ns

def triangulation(P1, P2, pts1, pts2):
  pts3D = np.empty((0,3))

  # Create skew-symmentric matrices
  for i in range(len(pts1)):
    ssm1 = np.zeros((3,3))
    ssm2 = np.zeros((3,3))

    ssm1[0][1] = -1.0
    ssm1[0][2] = pts1[i][1]
    ssm1[1][0] = 1.0
    ssm1[1][2] = -1*pts1[i][0]
    ssm1[2][0] = -1*pts1[i][1]
    ssm1[2][1] = pts1[i][0]

    ssm2[0][1] = -1.0
    ssm2[0][2] = pts2[i][1]
    ssm2[1][0] = 1.0
    ssm2[1][2] = -1*pts2[i][0]
    ssm2[2][0] = -1*pts2[i][1]
    ssm2[2][1] = pts2[i][0]

    # Multiply ssms with camera projection matrices
    tmp_1 = np.matmul(ssm1, P1)
    tmp_2 = np.matmul(ssm2, P2)
    A = np.empty((0,4))
    A = np.append(A, [tmp_1[0]], axis=0)
    A = np.append(A, [tmp_1[1]], axis=0)
    A = np.append(A, [tmp_2[0]], axis=0)
    A = np.append(A, [tmp_2[1]], axis=0)

    try:
      ns = nullspace(A)
      x = ns[0]/ns[3]
      y = ns[1]/ns[3]
      z = ns[2]/ns[3]

      if (x < 80 and y < 80 and z < 80):
        nsp = np.array([x,y,z])
        pts3D = np.append(pts3D, [nsp], axis=0)
    except:
      print("null space not found")


  return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
  # TO DO
  # last row of R is camera out vector

  # subtract x from c (cam loc)  (x-c)
  # mult with Rz.T 
  # if > 0, point is in front of camera

  print("\ndisambiguate_pose")

  best_pos = -1
  max_pts = 0

  # must satisfy for both cameras to count
  for i in range(len(Rs)):
    r_x = Rs[i][2]
    r_x = np.array([[r_x[0]], [r_x[1]], [r_x[2]]])
    r_x = r_x.reshape((1,3))
    c_p = Cs[i]
    cnt_pts = 0
    tmp_pts = pts3Ds[i]

    for j in range(len(tmp_pts)):
      pts = tmp_pts[j]
      pts = np.array([[pts[0]],[pts[1]],[pts[2]]])


      tmp1 = (np.subtract(pts, c_p))
      tmp = np.dot(r_x, tmp1)
      if (tmp > 0.0):
        cnt_pts += 1

    print(i)
    print(cnt_pts)
    
    if (cnt_pts > max_pts):
      max_pts = cnt_pts
      best_pos = i
  
  print("best idx: ")
  print(i)
  print("max: ")
  print(max_pts)

  R = Rs[i]
  C = Cs[i]
  pts3D = pts3Ds[i]
  
  return R, C, pts3D


def compute_rectification(K, R, C):
  # TO DO
  return H1, H2


def dense_match(img1, img2):
  # TO DO
  return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
  E = K.T @ F @ K
  R_1, R_2, t = cv2.decomposeEssentialMat(E)
  # 4 cases
  R1, t1 = R_1, t
  R2, t2 = R_1, -t
  R3, t3 = R_2, t
  R4, t4 = R_2, -t

  Rs = [R1, R2, R3, R4]
  ts = [t1, t2, t3, t4]
  Cs = []
  for i in range(4):
    Cs.append(-Rs[i].T @ ts[i])
  return Rs, Cs


def visualize_img_pair(img1, img2):
  img = np.hstack((img1, img2))
  if img1.ndim == 3:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  else:
    plt.imshow(img, cmap='gray')
  plt.axis('off')
  plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
  assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
  img_h = img1.shape[0]
  scale_factor1 = img_h/img1.shape[0]
  scale_factor2 = img_h/img2.shape[0]
  img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
  img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
  pts1 = pts1 * scale_factor1
  pts2 = pts2 * scale_factor2
  pts2[:, 0] += img1_resized.shape[1]
  img = np.hstack((img1_resized, img2_resized))
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  for i in range(pts1.shape[0]):
    plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
  plt.axis('off')
  plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
  assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
  ax1 = plt.subplot(121)
  ax2 = plt.subplot(122)
  ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
  ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

  for i in range(pts1.shape[0]):
    x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
    ax1.scatter(x1, y1, s=5)
    p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

  for i in range(pts2.shape[0]):
    x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
    ax2.scatter(x2, y2, s=5)
    p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

  ax1.axis('off')
  ax2.axis('off')
  plt.show()


def find_epipolar_line_end_points(img, F, p):
  img_width = img.shape[1]
  el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
  p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
  _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
  return p1, p2


def visualize_camera_poses(Rs, Cs):
  assert(len(Rs) == len(Cs) == 4)
  fig = plt.figure()
  R1, C1 = np.eye(3), np.zeros((3, 1))
  for i in range(4):
    R2, C2 = Rs[i], Cs[i]
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    draw_camera(ax, R1, C1)
    draw_camera(ax, R2, C2)
    set_axes_equal(ax)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(azim=-90, elev=0)
  fig.tight_layout()
  plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
  assert(len(Rs) == len(Cs) == 4)
  fig = plt.figure()
  R1, C1 = np.eye(3), np.zeros((3, 1))
  for i in range(4):
    R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    draw_camera(ax, R1, C1, 5)
    draw_camera(ax, R2, C2, 5)
    ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
    set_axes_equal(ax)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.view_init(azim=-90, elev=0)
  fig.tight_layout()
  plt.show()


def draw_camera(ax, R, C, scale=0.2):
  axis_end_points = C + scale * R.T  # (3, 3)
  vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
  vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

  # draw coordinate system of camera
  ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
  ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
  ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

  # draw square window and lines connecting it to camera center
  ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
  ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
  ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
  ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
  ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
  x_limits = ax.get_xlim3d()
  y_limits = ax.get_ylim3d()
  z_limits = ax.get_zlim3d()

  x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
  y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
  z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

  plot_radius = 0.5*max([x_range, y_range, z_range])

  ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
  ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
  ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
  plt.imshow(disparity, cmap='jet')
  plt.show()


if __name__ == '__main__':
  # read in left and right images as RGB images
  img_left = cv2.imread('./left.bmp', 1)
  img_right = cv2.imread('./right.bmp', 1)
  visualize_img_pair(img_left, img_right)

  # Step 1: find correspondences between image pair
  pts1, pts2 = find_match(img_left, img_right)
  visualize_find_match(img_left, img_right, pts1, pts2)

  # Step 2: compute fundamental matrix
  F = compute_F(pts1, pts2)
  visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

  # Step 3: computes four sets of camera poses
  K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
  Rs, Cs = compute_camera_pose(F, K)
  visualize_camera_poses(Rs, Cs)

  # Step 4: triangulation
  pts3Ds = []
  P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
  for i in range(len(Rs)):
    P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
    pts3D = triangulation(P1, P2, pts1, pts2)
    pts3Ds.append(pts3D)
  visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

  # Step 5: disambiguate camera poses
  R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

  # Step 6: rectification
  H1, H2 = compute_rectification(K, R, C)
  img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
  img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
  visualize_img_pair(img_left_w, img_right_w)

  # Step 7: generate disparity map
  img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
  img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
  img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
  img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
  disparity = dense_match(img_left_w, img_right_w)
  visualize_disparity_map(disparity)

  # save to mat
  sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                    'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
