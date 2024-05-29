import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():

    img_0, img_1, image_size = take_images(0,2)

    cv_file = cv2.FileStorage('stereo_calibration.yaml', cv2.FILE_STORAGE_READ)

    camera_matrix_0 = cv_file.getNode('camera_matrix_0').mat()
    dist_coeffs_0 = cv_file.getNode('dist_coeffs_0').mat()
    camera_matrix_1 = cv_file.getNode('camera_matrix_1').mat()
    dist_coeffs_1 = cv_file.getNode('dist_coeffs_1').mat()
    R = cv_file.getNode('R').mat()
    T = cv_file.getNode('T').mat()
    cv_file.release()

    print("camera_matrix_0 : ",camera_matrix_0)
    print("dist_coeffs_0 : ",dist_coeffs_0)
    print("camera_matrix_1 : ",camera_matrix_1)
    print("dist_coeffs_0 : ",dist_coeffs_1)
    print("R",R)
    print("T",T)
    
    R0, R1, P0, P1, Q, validPixROT0, validPixROT1 = cv2.stereoRectify(camera_matrix_0,dist_coeffs_0,camera_matrix_1,dist_coeffs_1,image_size, R,T,alpha=0)
    
    map0x, map0y = cv2.initUndistortRectifyMap(camera_matrix_0, dist_coeffs_0, R0, P0, image_size, cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, image_size, cv2.CV_32FC1)
    
    rectified_img_0 = cv2.remap(img_0, map0x, map0y, cv2.INTER_LINEAR)
    rectified_img_1 = cv2.remap(img_1, map1x, map1y, cv2.INTER_LINEAR)

    plt.figure(figsize=(20,10))
    plt.title("rectified_img_0")
    plt.imshow(rectified_img_0)
    plt.colorbar()
    plt.savefig("rectified_img_0.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    plt.figure(figsize=(20,10))
    plt.title("rectified_img_1")
    plt.imshow(rectified_img_1)
    plt.colorbar()
    plt.savefig("rectified_img_1.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    check_characteristic_point(rectified_img_0,rectified_img_1)

    stereo = cv2.StereoSGBM_create(minDisparity=200, numDisparities=16*9, blockSize=3, P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)

    disparity_map = stereo.compute(rectified_img_0, rectified_img_1).astype(np.float32) / 16.0
    disparity_map = cv2.medianBlur(disparity_map,5)
    
    print('image shape:',img_0.shape[:2])
    print('Disparity map shape:',disparity_map.shape)
    
    desplay_disparity_map(disparity_map.copy())
    
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    
    points_3D[points_3D == float('inf')] = 0
    points_3D[points_3D == float('-inf')] = 0

    depth_map = np.where(disparity_map > disparity_map.min(), points_3D[:, :,2],0)

    display_depth_map(depth_map)
    
    abs_z = np.abs(points_3D[:,:,2])
    
    nonzero_points = points_3D[abs_z != 0]

    min_distance_index = np.argmin(abs_z[abs_z != 0])
    nearest_nonzero_point = nonzero_points[min_distance_index]

    print(f'Nearest NonZero Point: {nearest_nonzero_point} mm')
    
    display_3d_map(points_3D[:, :,0],points_3D[:, :,1],points_3D[:, :,2])
    
def take_images(camera0_id,camera1_id):

    try:
        print(f'start to take a image camera_{camera0_id}')
        print(f'start to take a image camera_{camera1_id}')

        cap_0 = cv2.VideoCapture(camera0_id)
        cap_1 = cv2.VideoCapture(camera1_id)

        image_count = 0

        image_list = []

        target_img_0 = None
        target_img_1 = None
        
        retryNum = 0
        
        while True:
            ret_0, frame_0 = cap_0.read()
            ret_1, frame_1 = cap_1.read()
            
            if not ret_0 or not ret_1:
                retryNum += 1
                if retryNum>10:
                    break
                continue

            cv2.imshow('Frame0', frame_0)
            cv2.imshow('Frame1', frame_1)

            key = cv2.waitKey(1)

            if key == ord('s'):

                height_0, width_0 = frame_0.shape[:2]
                height_1, width_1 = frame_1.shape[:2]
                
                if (height_0, width_0) != (height_1, width_1):
                    print('camera 0 and camera 1 have different sizes')
                    continue
                else:
                    target_img_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
                    target_img_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
                    break
                
            elif key == ord('q'):
                break

        cap_0.release()
        cap_1.release()

        cv2.destroyAllWindows()

        plt.figure(figsize=(20,10))
        plt.title("target_0")
        plt.imshow(target_img_0)
        plt.colorbar()
        plt.savefig("target_0.png")
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        plt.figure(figsize=(20,10))
        plt.title("target_1")
        plt.imshow(target_img_1)
        plt.colorbar()
        plt.savefig("target_1.png")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
       
        return (target_img_0, target_img_1, target_img_0.shape[::-1])
        
    except Exception as e:
        print(f' take_images error: {e}')
        raise ValueError('take_images error')
    
def check_characteristic_point(rectified_img_0,rectified_img_1):

    sift = cv2.SIFT_create()

    keypoints_0, descriptors_0 = sift.detectAndCompute(rectified_img_0, None)
    keypoints_1, descriptors_1 = sift.detectAndCompute(rectified_img_1, None)

    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
    search_params = dict(check=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_0,descriptors_1,k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)
    
    #orb = cv2.ORB_create()

    #keypoints_0, descriptors_0 = orb.detectAndCompute(rectified_img_0, None)
    #keypoints_1, descriptors_1 = orb.detectAndCompute(rectified_img_1, None)

    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #matches = bf.match(descriptors_0, descriptors_1)

    #matches = sorted(matches, key=lambda x:x.distance)

    matched_image = cv2.drawMatches(rectified_img_0, keypoints_0, rectified_img_1, keypoints_1, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(20,10))
    plt.title("matched image")
    plt.imshow(matched_image)
    plt.colorbar()
    plt.savefig("matched_image.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close()
    
    points_0 = np.array([keypoints_0[m.queryIdx].pt for m in good_matches],dtype=np.float32)
    points_1 = np.array([keypoints_1[m.trainIdx].pt for m in good_matches] ,dtype=np.float32)

    diff = np.abs(points_0[:,1] - points_1[:,1])

    print('Row diff:')
    print('Mean: ',np.mean(diff))
    print('Median: ',np.median(diff))
    print('Max: ',np.max(diff))

    tolerance = 1.0
    num_within_tolerance = np.sum(diff <= tolerance)

    print(f'Numer of matches within {tolerance} pixels tolerance : ',num_within_tolerance)
    print(f'Total numer of matches : ',len(matches))
    
def desplay_disparity_map(disparity_map):

    try:

        disparity_map_filtered = disparity_map.copy()
        disparity_map_filtered[disparity_map == 0] = 255

        plt.figure(figsize=(20,10))
        plt.title("disparity map")
        plt.imshow(disparity_map_filtered)
        plt.colorbar()
        plt.savefig("disparity.png")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        
    except Exception as e:
        print(f'[desplay_disparity_map][error] {e}')

def display_depth_map(depth):

    plt.figure(figsize=(20,10))
    plt.title("depth map")
    plt.imshow(depth,cmap='jet')
    plt.colorbar()
    plt.savefig("depth.png")
    plt.show()
    #plt.show(block=False)
    #plt.pause(5)
    #plt.close()

def display_3d_map(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X,Y,Z, c=Z, cmap='viridis',marker='o')

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    plt.savefig("3d_map.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

if __name__ == "__main__":
    main()
