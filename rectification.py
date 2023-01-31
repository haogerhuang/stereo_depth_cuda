import cv2
import numpy as np
from calibration import feature_matching, estimate_F_RANSAC
from util import Dataset, toHomogeneous
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='curule', choices=['curule', 'octagon', 'pendulum'])
    args = parser.parse_args()

    print('Dataset:', args.name)

    dataset = Dataset(args.name)
    img0 = dataset.img0
    img1 = dataset.img1

    h,w,_ = img0.shape
    
    # Find matching points from stereo images 
    pnts1, pnts2, matches, kp0, kp1 = feature_matching(img0, img1)
    
    # RANSAC to estimate fundamental matrix
    F, mask = estimate_F_RANSAC(pnts1, pnts2, 32,  2500, 0.001)
    
    # Find homography to rectify the images
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pnts1[mask], pnts2[mask], F, imgSize=(w, h))
    rect_img0 = cv2.warpPerspective(img0, H1, (w,h))
    rect_img1 = cv2.warpPerspective(img1, H2, (w,h))
    

    print('Homography for the left image')
    print(H1)

    print('Homography for the right image')
    print(H2)

    #+++++++++++++++For Visualization++++++++++++++++++++++++

    warp_pnts1 = (H1 @ toHomogeneous(pnts1[mask]).T).T
    warp_pnts1 = warp_pnts1/warp_pnts1[:,2].reshape(-1,1)

    warp_pnts2 = (H2 @ toHomogeneous(pnts2[mask]).T).T
    warp_pnts2 = warp_pnts2/warp_pnts2[:,2].reshape(-1,1)

    rand_idx = np.random.choice(np.arange(warp_pnts1.shape[0]), 50, replace=False)
    selected_idx = []
    
    # Random choose up to 10 points for visualization
    # The chosen points have to be 0 < x < w, 0 < y < h to be visualized
    i = 0
    while len(selected_idx) < 10 and i < warp_pnts1.shape[0]:
        idx = rand_idx[i]
        pnt1 = warp_pnts1[idx]
        pnt2 = warp_pnts2[idx]
        if pnt2[0] >= 0 and pnt2[0] < w and pnt2[1] >= 0 and pnt2[1] < h: 
            selected_idx.append(idx)
        i += 1

    selected_idx = np.array(selected_idx).astype(int)

    pnts1 = warp_pnts1[selected_idx]
    pnts2 = warp_pnts2[selected_idx]

    # Epipolar lines
    l2 = (F@pnts1.T).T
    l1 = (F.T@pnts2.T).T


    for i in range(pnts1.shape[0]):
        x1 = 0
        x2 = w-1
        pnt2 = pnts2[i]

        # Plot epipolar line and the corresponding key points
        a,b,c = l2[i]

        cv2.line(rect_img1, (x1,int((-a*x1-c)/b)), (x2,int((-a*x2-c)/b)), (0,255,0), 4)
        cv2.circle(rect_img1, (int(pnt2[0]), int(pnt2[1])), 7, (0,255,0), -1)

        pnt1 = pnts1[i]
        # Plot epipolar line and the corresponding key points
        a,b,c = l1[i]

        cv2.line(rect_img0, (x1,int((-a*x1-c)/b)), (x2,int((-a*x2-c)/b)), (0,255,0), 4)
        cv2.circle(rect_img0, (int(pnt1[0]), int(pnt1[1])), 7, (0,255,0), -1)


    res_img = np.concatenate([rect_img0, rect_img1], 1)
    warp_pnts2[:,0] += w 

    cv2.imwrite(dataset.name+'_epipolarlines.png', res_img)
    cv2.imshow('Epipolar Lines', res_img)
    cv2.waitKey(0)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




