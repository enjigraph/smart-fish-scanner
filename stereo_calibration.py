import single_image
import calibration_by_chessboard
import cv2
import numpy as np

def main():

    chessboard_size = (7,10)
    square_size_mm = 24

    objp = np.zeros(( chessboard_size[0]*chessboard_size[1], 3 ), np.float32 )
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2) * square_size_mm

    objpoints = []
    imgpoints_0 = []    
    imgpoints_1 = []    
        
    single_image.take(0,'./calibration_images/single/camera_0')
    single_image.take(2,'./calibration_images/single/camera_1')

    camera_matrix_0, dist_coeffs_0 = calibration_by_chessboard.calibration((7,10),24,'calibration_0.yaml', './calibration_images/single/camera_0')
    camera_matrix_1, dist_coeffs_1 = calibration_by_chessboard.calibration((7,10),24,'calibration_1.yaml', './calibration_images/single/camera_1')
    
    image_list = take_images(0,2)
    
    used_images_num = 0

    image_size = (0,0)
    
    for image in image_list:
        img_0 = cv2.imread(f'./calibration_images/stereo/camera_0/{image}')
        img_1 = cv2.imread(f'./calibration_images/stereo/camera_1/{image}')

        height_0, width_0 = img_0.shape[:2]
        height_1, width_1 = img_1.shape[:2]

        if (height_0, width_0) != (height_1, width_1):
            print('camera 0 and camera 1 have different sizes')
            continue
        else:
            image_size = (height_0, width_0)
        
        gray_0 = cv2.cvtColor(img_0,cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
        
        ret_0, centers_0 = cv2.findChessboardCorners(gray_0,chessboard_size,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        ret_1, centers_1 = cv2.findChessboardCorners(gray_1,chessboard_size,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        print(f'{image} : {ret_0}, {ret_1}')
        
        if not ret_0 or not ret_1:
            continue
        
        used_images_num += 1
        
        objpoints.append(objp)
        
        sub_pixel_centers_0 = cv2.cornerSubPix(gray_0, centers_0, (5,5), (-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        sub_pixel_centers_1 = cv2.cornerSubPix(gray_1, centers_1, (5,5), (-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        imgpoints_0.append(sub_pixel_centers_0)
        imgpoints_1.append(sub_pixel_centers_1)
        
        cv2.imshow('img_0',cv2.drawChessboardCorners(img_0, chessboard_size, sub_pixel_centers_0, ret_0))
        cv2.imshow('img_1',cv2.drawChessboardCorners(img_1, chessboard_size, sub_pixel_centers_1, ret_1))
        cv2.waitKey(500)

    print('Used images size :',image_size)
    print('Used images num :',used_images_num)

    ret, optimized_camera_matrix_0, optimized_dist_coeffs_0, optimized_camera_matrix_1, optimized_dist_coeffs_1, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints_0,imgpoints_1,camera_matrix_0, dist_coeffs_0,camera_matrix_1, dist_coeffs_1,gray_0.shape[::-1],criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30, 1e-6),flags=cv2.CALIB_FIX_INTRINSIC)

    print("R",R)
    print("T",T)
    print("E",E)
    print("F",F)

    cv_file = cv2.FileStorage('stereo_calibration.yaml', cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix_0", camera_matrix_0)
    cv_file.write("dist_coeffs_0",dist_coeffs_0)
    cv_file.write("camera_matrix_1", camera_matrix_1)
    cv_file.write("dist_coeffs_1",dist_coeffs_1)
    cv_file.write("R",R)
    cv_file.write("T",T)
    cv_file.write("E",E)
    cv_file.write("F",F)
    cv_file.release()
    
    
def take_images(camera0_id,camera1_id):

    try:
        print(f'start to take a image camera_{camera0_id}')
        print(f'start to take a image camera_{camera1_id}')

        cap0 = cv2.VideoCapture(camera0_id)
        cap1 = cv2.VideoCapture(camera1_id)

        image_count = 0

        image_list = []
        
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            
            if not ret0 or not ret1:
                break

            cv2.imshow('Frame0', frame0)
            cv2.imshow('Frame1', frame1)

            key = cv2.waitKey(1)

            if key == ord('s'):
                cv2.imwrite(f'./calibration_images/stereo/camera_0/image_{image_count}.png',frame0)
                cv2.imwrite(f'./calibration_images/stereo/camera_1/image_{image_count}.png',frame1)
                image_list.append(f'image_{image_count}.png')
                print(f'Image {image_count} saved.')
                image_count += 1
            elif key == ord('q'):
                break

        cap0.release()
        cap1.release()

        cv2.destroyAllWindows()

    except:
        print(f'error: camera_{camera_id} is not found')

    return image_list

def calibrate_single_camera(objpoints,imgpoints,gray):

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print('Image shape :',gray.shape[::-1])
    print('RMS :',ret)
    print("カメラ行列：\n",camera_matrix)
    print("歪み係数：\n",dist_coeffs)

    return (ret, camera_matrix, dist_coeffs, rvecs, tvecs)

    
    
if __name__ == "__main__":
    main()
