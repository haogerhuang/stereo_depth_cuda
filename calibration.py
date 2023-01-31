import cv2
import numpy as np
from util import Dataset, toHomogeneous
import argparse


def feature_matching(img0, img1, ratio=0.6):
    """
    Function that detect features and match features 
    between two given images
    1. SIFT feature detection
    2. Match features using Brute-Force
    3. Filter matched features using the ratio between the top-2 matches
    
    Input:
    - img0 : Left image
    - img1 : Right image
    - ratio: The ratio threshold between the top and second matching distance
    Return:
    - pnts1: Filtered matching feature point positions in the left image
    - pnts2: Filtered matching feature point positions in the right image
    - filter_matches: Filtered matches, [Dmatches]
    - kp1: Filtered matching key points in the left image
    - kp2: Filtered matching key points in the right image
    """
    sift = cv2.SIFT_create()
    # Detect SIFT features for both images
    kp1, des1 = sift.detectAndCompute(img0,None)
    kp2, des2 = sift.detectAndCompute(img1,None)
    
    # Find top-2 matches using brute force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    pnts1 = []
    pnts2 = []
    filter_matches = []
    # Filter matches
    for m1, m2 in matches:
        # Keep matches when the distance 
        # between the 1 and 2 match are larger than a threhold
        if m1.distance < ratio*m2.distance:
            pnts1.append(kp1[m1.queryIdx].pt)
            pnts2.append(kp2[m1.trainIdx].pt)
            filter_matches.append(m1)

    pnts1 = np.array(pnts1)       
    pnts2 = np.array(pnts2)       


    return pnts1, pnts2, filter_matches, kp1, kp2

def estimate_fundamental_matrix(pnts1, pnts2):
    """
    Estimate the fundamental matrix given two set of matching points.
    1. Point normalization.
    2. Solve pnts2.T @ F @ pnts1 = 0 
    3. Correct F

    Input: 
    - pnts1: Points form left image in [pixels, pixels] 
    - pnts2: Points form right image in [pixels, pixels]

    Return:
    - F: Fundamental Matrix
    """

    N = pnts1.shape[0]

    if pnts1.shape[1] < 3:
        pnts1 = toHomogeneous(pnts1)
        pnts2 = toHomogeneous(pnts2)

    # ======================== [Optional] ========================
    # ====Point normalization to produce more stable estimation===
    C1 = np.eye(3)
    C2 = np.eye(3)

    C1[0:2,2] = -pnts1[:,0:2].mean(0)
    C2[0:2,2] = -pnts2[:,0:2].mean(0)

    S1 = np.eye(3)
    S2 = np.eye(3)

    std1 = (pnts1[:,0:2] - pnts1[:,0:2].mean(0).reshape(1,-1)).std(0).sum()
    std2 = (pnts2[:,0:2] - pnts2[:,0:2].mean(0).reshape(1,-1)).std(0).sum()

    S1[0,0] = (2**0.5)/std1
    S1[1,1] = (2**0.5)/std1

    S2[0,0] = (2**0.5)/std2
    S2[1,1] = (2**0.5)/std2

    T1 = S1 @ C1
    T2 = S2 @ C2


    pnts1 = (T1 @ pnts1.T).T
    pnts2 = (T2 @ pnts2.T).T

    # ===============================================================

    pnts1 = pnts1.reshape((N,3,1))
    pnts2 = pnts2.reshape((N,1,3))

    A = pnts1 @ pnts2
    A = A.reshape((N,9))

    U,S,V = np.linalg.svd(A)
    F = V[-1]
    
    F = F.reshape(3,3).T
    
    # Correct the fundamental matrix
    F = T2.T @ F @ T1

    # ------Sanity Check -------
    #test = pnts2 @ F @ pnts1
    #test = test.reshape(N)
    #print(test)
    
    return F


def estimate_F_RANSAC(pnts1, pnts2, samples, iterations, thres=0.01):
    """
    Use RANSAC to estimate fundamental matrix
    Input:
    - pnts1: Points from left image
    - pnts2: Points from right image
    - samples: Number of points to be sampled for each iteration
    - iterations: Total iteration
    - thres: Threshold for inliers
    Return:
    - F: Fundamental Matrix
    - mask: Mask indicating whether a pair of matching points falls into the inlier
    """

    N = pnts1.shape[0]
   
    # Convert to homogeneous coordinate
    pnts1_h = toHomogeneous(pnts1)
    pnts2_h = toHomogeneous(pnts2)

    pnts1_1 = pnts1_h.reshape((N,3,1))
    pnts2_1 = pnts2_h.reshape((N,1,3))
    
    best_score = 0
    best_F = None

    for _ in range(iterations):
        # Random sample [samples] points
        idx = np.random.choice(np.arange(N), samples, replace=False)
        # Estimate fundamental matrix 
        F = estimate_fundamental_matrix(pnts1_h[idx], pnts2_h[idx])

        score = pnts2_1 @ F @ pnts1_1
        
        # Compute the x'Fx and see how many are below threshold
        score = (np.abs(score.reshape(N)) < thres).sum()
        
        # Save F if the most inliers found in this iteration
        if score > best_score:
            best_score = score
            best_F = F

    # Obtain mask 
    mask = pnts2_1 @ F @ pnts1_1
         
    mask = np.abs(mask.reshape(N)) < thres
    
    return best_F, mask


    
def estimate_essential_matrix(F, K0, K1):
    """
    Estimate essential matrix given fundamental matrix and camera intrinsic matrix
    Input:
    - F: Fundamental Matrix 
    - K: Camera Intrinsic Matrix
    Return:
    - E: Essential Matrix
    """
    E = K1.T @ F @ K0
    U, S, V = np.linalg.svd(E)
    # Correct E using [1,1,0] as singular values
    E = U @ np.diag(np.array([1,1,0])) @ V
    return E

def estimate_camera_pose(E, pnts1, pnts2, K1, K2):
    """
    Recover camera pose from essential matrix
    1. Four poses (R1, C1), (R2, C2), (R3, C3), (R4, C4) are found using SVD
    2. Perform triangulation on each points using each poses
    3. Use the Cheirality Condition to determine which one is the correct pose
    """
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    
    # Translation
    C1 = U[:,-1]
    # Rotation
    R1 = U @ W @ V

    C2 = -C1
    R2 = R1
    C3 = C1
    R3 = U @ W.T @ V
    C4 = -C3
    R4 = R3

    # Flip sign if the det(R) is 0
    

    if np.linalg.det(R1) == -1:
        C1 = -C1
        R1 = -R1
        C2 = -C2
        R2 = -R2
    if np.linalg.det(R3) == -1:
        C3 = -C3
        R3 = -R3
        C4 = -C4
        R4 = -R4

    Poses = [(R1, C1), (R2, C2), (R3, C3), (R4, C4)]

    # P = K[R | -RC]
    P0 = get_projection_matrix(K1, np.eye(3), np.zeros(3))

    CCs = []
    for R, C in Poses:
        P = get_projection_matrix(K2, R, C)

        # Triangulation and obtain 3D points
        X = triangulation(P0, P, pnts1, pnts2)

        # Cheirality Condition 
        che = cheirality_condition(X[:,:3], np.eye(3), np.zeros(3), R, C)
        CCs.append(che)

    CCs = np.array(CCs)
    # Get the pose with the maximum points that satisfy the Cheirality Condition
    R, C = Poses[CCs.argmax()]
    return R, C
    

def cheirality_condition(X, R1, C1, R2, C2):
    """
    Check if the given 3D points from triangulation satisfy the cheirality condition
    Input:
    - X: 3D points
    - R1: Rotation
    - C1: Translation
    - R2: Rotation
    - C2: Translation
    """
    # Get mask of points that satisfied cheirality condition for the first pose
    satisfy1 = ((R1[-1] @ (X - C1).T).T) > 0

    # Get mask of points that satisfied cheirality condition for the second pose

    satisfy2 = ((R2[-1] @ (X - C2).T).T) > 0
    
    # Return number of points that satisfied cheirality condition for both poses
    return (satisfy1 & satisfy2).astype(int).sum() 


def get_projection_matrix(K, R, C):
    """
    Get projection matrix P from camera intrinsic matrix, rotation and translation
    P = K[R | -RC]
    Input:
    - K: Camera Intrinsic Matrix
    - R: Rotation
    - C: Translation
    Return:
    - P: Projection Matrix
    """ 
    return K @ np.concatenate([R, -R @ C.reshape(-1,1)], 1)    

def triangulation(P1, P2, pnts1, pnts2):
    """
    Function that triangulate the points given projection matrixs and points
    Input:
    - P1: Projection Matrix from the first pose 
    - P2: Projection Matrix from the second pose
    - pnts1: Points from the first image [Pixels]
    - pnts2: Points from the first image [Pixels]
    Return:
    - X: 3D points
    """
    N = pnts1.shape[0]
    X = []
    for i in range(N):
        x1, y1 = pnts1[i]
        x2, y2 = pnts2[i]
        
        A = np.array([y1*P1[2].T - P1[1].T,
                          P1[0].T - x1*P1[2].T,
                          y2*P2[2].T - P2[1].T,
                          P2[0].T - x2*P2[2].T])
        U, S, V = np.linalg.svd(A.T @ A)
        x = V[-1]
        x /= x[-1]
        X.append(x)
    
    X = np.array(X)
    return X

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='curule', choices=['curule', 'octagon', 'pendulum'])
    args = parser.parse_args()
    
    dataset = Dataset(args.name)

    print('Dataset:', args.name)
    
    img0 = dataset.img0
    img1 = dataset.img1

    K1 = dataset.calib.cam0
    K2 = dataset.calib.cam1

    # SIFT feature matching
    pnts1, pnts2, matches, kp0, kp1 = feature_matching(img0, img1)
    
    # ++++++++++++++++++ For Visualization ++++++++++++++++++++++
    # Randomly sample 100 matches and plot
    rand_idx = np.random.choice(np.arange(pnts1.shape[0]), 100, replace=False)
    rand_pnts1 = pnts1[rand_idx]
    rand_pnts2 = pnts2[rand_idx]

    cat_img = np.concatenate([img0, img1], 1)
    h,w,_ = img0.shape

    for pnt1, pnt2 in zip(rand_pnts1, rand_pnts2):
        cv2.circle(cat_img, (int(pnt1[0]), int(pnt1[1])), 2, (255,0,0), 2)
        cv2.circle(cat_img, (int(pnt2[0]+w), int(pnt2[1])), 2, (255,0,0), 2)
        cv2.line(cat_img, (int(pnt1[0]),int(pnt1[1])), (int(pnt2[0]+w),int(pnt2[1])), 
                (255,0,0), 2)

    cv2.imwrite(dataset.name+'_feature_match.jpg', cat_img)
 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   

    # Fundamental Matrix estimation using RANSAC
    F, mask = estimate_F_RANSAC(pnts1, pnts2, 32, 2000, 0.001)
    print('Estimate Fundamental Matrix:')
    print(F)
    print('Number of inliers:', mask.sum())

    # Essential Matrix estimation
    E = estimate_essential_matrix(F, K1, K2)

    print('Essential Matrix:')
    print(E)

    pnts1_norm = (np.linalg.inv(K1) @ toHomogeneous(pnts1).T).T
    pnts2_norm = (np.linalg.inv(K2) @ toHomogeneous(pnts2).T).T
    pnts1_norm = pnts1_norm[:,0:2] / pnts1_norm[:,-1].reshape(-1,1)
    pnts2_norm = pnts2_norm[:,0:2] / pnts2_norm[:,-1].reshape(-1,1)

    R, C = estimate_camera_pose(E, pnts1[mask], pnts2[mask], K1, K2)

    print('Rotation:')
    print(R)
    print('Translation:')
    print(C)

