import cv2
import numpy as np
from calibration import feature_matching, estimate_F_RANSAC
from util import Dataset, toHomogeneous
from tqdm import tqdm
import argparse


def disparity(img0, img1, kernel_size, ndisp):
    """
    Compute disparity
    Input:
    - img0: Rectified left image
    - img1: Rectified right image
    - kernel_size: window size to compute SSD
    - ndisp: Conservative bound of the maximum disparity
    Return:
    - Disparity: Result disparity with size same as img0
    """

    h,w = img0.shape[0:2]
    Disparity = np.zeros([h,w])
    
    # indices_i: [[0],[1],[2],[3],....], 
    # indices_j: [[0,1,2,...,ndisp-1],[1,2,3,...,ndsip],[2,3,4,....,ndisp+1]]
    # These indices are used to get the left most [ndsip] SSD values for each 
    # kernel locations
    indices = np.arange(0, ndisp).reshape(-1,1)
    # N x ndisp
    indices_j = np.clip(indices + np.arange(0, w-kernel_size+1), 0, w-kernel_size).T
    # N x 1
    indices_i = np.arange(indices_j.shape[0]).reshape(-1,1)
    
    img0 = img0.astype(float)
    img1 = img1.astype(float)
   
    # Kernel indices:
    # e.g. if kernel size is 3
    # The kernel_indices will look like:
    # [[0,1,2],[1,2,3],[2,3,4],...] 
    # The purpose is to get all the values in each row of all kernels 
    # at the same time
    kernel_indices = np.arange(0, kernel_size).reshape(-1,1)
    kernel_indices = kernel_indices + np.arange(0, w-kernel_size+1)
    kernel_indices = kernel_indices.T
    
    #idx that tracks the row index to be removed in each iteration
    idx = 0
    SSD = None
    for i in tqdm(range(kernel_size//2, h-kernel_size//2)):
        # First iteration, which SSD is not initialized yet
        if SSD is None:
            kernels0 = []
            kernels1 = []
            for j in range(kernel_size):
                # For each row in the first kernel 
                kernels0.append(np.expand_dims(img0[j][kernel_indices], 1))
                kernels1.append(np.expand_dims(img1[j][kernel_indices], 1))
    
            # All kernels in the first row
            # Where N is the number of kernels each row
            kernels0 = np.concatenate(kernels0, 1) # (N, K, K, 3)
            kernels1 = np.concatenate(kernels1, 1) # (N, K, K, 3)
    
            N = kernels0.shape[0]
    
            kernels0 = kernels0.reshape(N, -1) # (N, K*K*3)
            kernels1 = kernels1.reshape(N, -1) # (N, K*K*3)
    
            # Compute the SSD between each kernels in kernels0 and kernels1
            # Using: (A - B)^2 = A^2 + B^2 - 2AB
            # SSD: (N, N)
            SSD = (kernels0**2).sum(-1).reshape(-1,1) +\
                    (kernels1**2).sum(-1).reshape(1,-1) -\
                    2*(kernels0 @ kernels1.T)
    
        else:
            # When shift to next row, only the first row in the kernels are removed
            # And one row of kernels are added
            # Therefore SSD only needs to be updated 
            # by removing the SSD of the first row
            # and adding the SSD of the new row

            # SSD values to be removed
            remove0 = img0[idx][kernel_indices].reshape(N,-1)
            remove1 = img1[idx][kernel_indices].reshape(N,-1)
    
            # SSD values to be added
            add0 = img0[idx+kernel_size][kernel_indices].reshape(N, -1)
            add1 = img1[idx+kernel_size][kernel_indices].reshape(N, -1)
    
            # Update SSD
            SSD -= (remove0**2).sum(-1).reshape(-1,1) +\
                (remove1**2).sum(-1).reshape(1,-1) - 2*(remove0 @ remove1.T)
    
            SSD += (add0**2).sum(-1).reshape(-1,1) +\
                (add1**2).sum(-1).reshape(1,-1) - 2*(add0 @ add1.T)
    
            idx += 1
    
        # Get only the left most [ndisp] values of SSD
        _SSD = SSD[indices_i, indices_j] # (N, ndisp)
    
        # Getting the index with the min value gives the disparity
        disparity = _SSD.argmin(-1)
        Disparity[i,kernel_size//2:kernel_size//2+N] = disparity

    return Disparity

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='curule', choices=['curule', 'octagon', 'pendulum'])
    parser.add_argument('-k', '--kernel_size', default=9, type=int)
    args = parser.parse_args()



    print('Compute disparity and depth image of for the {} dataset.'.format(args.name))
    dataset = Dataset(args.name)

    if args.kernel_size % 2 == 0:
        raise Exception('Kernel size is {}. Kernel size shoud be odd'.format(args.kernel_size))
    print('Kernel size:', args.kernel_size)


    img0 = dataset.img0
    img1 = dataset.img1
    
    h,w = img0.shape[0:2]
    
    # Feature matching
    pnts1, pnts2, matches, kp0, kp1 = feature_matching(img0, img1)
    
    # Fundamental Matrix estimation
    F, mask = estimate_F_RANSAC(pnts1, pnts2, 32, 3000, 0.001)
    
    print('Number of Inliers of current Fundamental Matrix', mask.sum())
    
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pnts1[mask], pnts2[mask], F, imgSize=(w, h))
    rect_img0 = cv2.warpPerspective(img0, H1, (w,h))
    rect_img1 = cv2.warpPerspective(img1, H2, (w,h))
    
    focal_length = dataset.calib.cam0[0,0]*0.5 + dataset.calib.cam1[1,1]*0.5
    baseline = dataset.calib.baseline
    
    ndisp = dataset.calib.ndisp
    
    Disparity = disparity(rect_img0, rect_img1, args.kernel_size, ndisp)
    
    Disparity = Disparity.reshape(-1)
    
    # Mark the zero disparities to -1 to prevent dividing zeros
    # when computing depth
    m = Disparity == 0
    Disparity[m] = -1
    Disparity = Disparity.reshape(h, w)
    
    # Compute depth image
    Depth = baseline*0.001*focal_length/(Disparity)
    
    # Change Negative depths to 0
    Depth = np.maximum(Depth, 0)
    
    # Clip disparity values using the bounding values provided
    Disparity = np.clip(Disparity, dataset.calib.vmin, dataset.calib.vmax)
    
    # Normalize between 0~1
    Disparity = (Disparity - Disparity.min())/(Disparity.max()-Disparity.min())
    
    # 0 ~ 255
    gray_disparity = (Disparity*255).astype(np.uint8)
    # Convert to color heap map
    color_disparity = cv2.applyColorMap(gray_disparity, cv2.COLORMAP_JET)
    
    cv2.imwrite(dataset.name+'_disparity.png', gray_disparity)
    cv2.imwrite(dataset.name+'_clr_disparity.png', color_disparity)
    
    # Normalize between 0~1
    Depth = (Depth - Depth.min())/(Depth.max() - Depth.min())
    
    gray_depth = (Depth*255).astype(np.uint8)
    color_depth = cv2.applyColorMap(gray_depth, cv2.COLORMAP_JET)
    
    cv2.imwrite(dataset.name+'_depth.png', gray_depth)
    cv2.imwrite(dataset.name+'_clr_depth.png', color_depth)
