import cv2
import numpy as np
from matplotlib import pyplot as plt

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

    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16*9, blockSize=5, P1=8*3*5**2, P2=32*3*5**2, disp12MaxDiff=1, uniquenessRatio=15, speckleWindowSize=100, speckleRange=32)

    disparity_map = stereo.compute(rectified_img_0, rectified_img_1).astype(np.float32) / 16.0

    desplay_disparity_map(disparity_map)
    
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    depth_map = points_3D[:, :,2]

    max_depth = np.max(depth_map)

    print('max_depth: ',max_depth)

    if max_depth > 0:
        normalized_depth_map = depth_map / max_depth
    else:
        normalized_depth_map = depth_map

    cv2.imshow('Depth Map', normalized_depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.imshow(disparity_map, 'gray')
    plt.show()
    
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
        image_size = (0,0)

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
                    image_size = (height_0, width_0)
                    print('image size:',image_size)
                    target_img_0 = frame_0
                    target_img_1 = frame_1
                    break
                
            elif key == ord('q'):
                break

        cap_0.release()
        cap_1.release()

        cv2.destroyAllWindows()

        return (target_img_0, target_img_1, image_size)
        
    except Exception as e:
        print(f' take_images error: {e}')
        raise ValueError('take_images error')

def desplay_disparity_map(disparity_map):
    
    plt.figure(figsize=(13,3))
    plt.imshow(disparity_map)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(13,3))
    plt.imshow(disparity_map/16)
    plt.colorbar()
    plt.show()
    
    
if __name__ == "__main__":
    main()
