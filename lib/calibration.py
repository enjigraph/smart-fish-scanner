import cv2
import os
import serial
import time
import glob
import itertools
import numpy as np
import lib.utils as utils
from lib.camera import Camera

camera = Camera()

def take_images(folder_path,direction,adjustment):
        
    ser = serial.Serial('/dev/ttyACM2',9600)
    time.sleep(2)

    try:

        print(f'start to take a image camera')

        count = 0

        camera.on()

        angle = ['90,90','105,90','75,90','90,75','90,105','100,100','80,80']
        patterns = list(itertools.product(angle,direction))
        
        for pattern in patterns:

            if count % 5 == 0:
                camera.grab()

            instruction = ','.join(map(str,pattern))
            print(f'{count}: {instruction}')
            ser.write(instruction.encode())
            time.sleep(4)

            if not adjustment:
                ret, frame = camera.get_image()
                
                if not ret:
                    continue

                cv2.imwrite(f'{folder_path}/{count}.png',frame)
                count += 1

            else:
                for i in range(5):

                    ret, frame = camera.get_image()
                                    
                    if not ret:
                        continue
                    
                    adjustment_directions = get_missing_ar_marker_ids(frame)

                    if len(adjustment_directions) > 0:
                        for adjustment_direction in adjustment_directions:
                            print(f'ar is missing. adjustment_direction: {adjustment_direction}.')
                            parts = instruction.split(',')
                            parts[2]= adjustment_direction
                            new_instruction = ','.join(parts)
                            ser.write(new_instruction.encode())
                            time.sleep(2)
                            camera.grab()
                    else:
                        cv2.imwrite(f'{folder_path}/{count}.png',frame)
                        count += 1
                        break

        ser.write('90,90,stop'.encode())
        time.sleep(2)
        camera.release()
       
    except Exception as e: 
        print(f'take image error: {e}')

    ser.close()

def take_images_by_manual(folder_path):
    
    ser = serial.Serial('/dev/ttyACM2',9600)
    time.sleep(2)

    try:
        print(f'start to take a image camera')
        camera.on()

        count = sum(len(files) for _ , _, files in os.walk(folder_path))
        patterns = ['90,90,stop','105,90,stop','75,90,stop','90,75,stop','90,105,stop','100,100,stop','80,80,stop']

        for pattern in patterns:

            print(f'{count}: {pattern}')
            ser.write(pattern.encode())
            time.sleep(4)

            ret, frame = camera.get_image()
            
            if not ret:
                continue
                
            cv2.imwrite(f'{folder_path}/{count}.png',frame)
            count += 1

        ser.write('90,90,stop'.encode())
        time.sleep(2)
        camera.release()
    except Exception as e:
        print(f'take_images_by_manual error: {e}')

    ser.close()

def get_missing_ar_marker_ids(frame):

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary,parameters=parameters)

    frame_with_markers = cv2.aruco.drawDetectedMarkers(frame.copy(),corners,ids)

    print(ids)
    target_ids = {0, 1, 2, 3}

    detected_ids = set(ids.flatten())

    missing_ids = target_ids - detected_ids

    if 0 in missing_ids and 1 in missing_ids:
        return ['left']

    if 2 in missing_ids and 3 in missing_ids:
        return ['right']

    if 0 in missing_ids:
        return ['left','leftRotation']

    if 1 in missing_ids:
        return ['left','rightRotation']

    if 2 in missing_ids:
        return ['right','leftRotation']

    if 3 in missing_ids:
        return ['right','rightRotation']

    return []
    
    
def get_parameters(chessboard_size,square_size_mm,file_name,calibration_images_folder_path):

    objp = np.zeros(( chessboard_size[0]*chessboard_size[1], 3 ), np.float32 )
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2) * square_size_mm

    objpoints = []
    imgpoints = []

    images = glob.glob(f'{calibration_images_folder_path}/*.png')

    used_images_num = 0
    
    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ret, centers = cv2.findChessboardCorners(gray,chessboard_size,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        print(f'{image} : {ret}')
        
        if ret:
            used_images_num += 1
        
            objpoints.append(objp)

            sub_pixel_centers = cv2.cornerSubPix(gray, centers, (5,5), (-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            imgpoints.append(sub_pixel_centers)
            
            img = cv2.drawChessboardCorners(img, chessboard_size, sub_pixel_centers, ret)
            camera.show('img',img)
            
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None,None)

    print('Used images num :',used_images_num)
    print('Image shape :',gray.shape[::-1])
    print('RMS :',ret)
    print("カメラ行列：\n",camera_matrix)
    print("歪み係数：\n",dist_coeffs)
    
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        
    print("total error: {}".format(mean_error/len(objpoints)))

    cv_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", camera_matrix)
    cv_file.write("dist_coeffs",dist_coeffs)
    cv_file.release()

    #cv2.destroyAllWindows()

    return camera_matrix, dist_coeffs

def get_parameters_of_fisheye(chessboard_size,square_size_mm,file_name,calibration_images_folder_path):

    delete_image = None
    
    try:

        objp = np.zeros((1, chessboard_size[0]*chessboard_size[1], 3 ), np.float32 )
        objp[0,:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2) * square_size_mm

        objpoints = []
        imgpoints = []
        
        images = glob.glob(f'{calibration_images_folder_path}/*.png')
        
        used_images_num = 0
    
        for image in images:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blurred_gray = cv2.GaussianBlur(gray,(5,5),0)
        
            ret, centers = cv2.findChessboardCorners(blurred_gray,chessboard_size,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            print(f'{image} : {ret}')
        
            if ret:
                used_images_num += 1
                
                objpoints.append(objp)
                
                sub_pixel_centers = cv2.cornerSubPix(gray, centers, (5,5), (-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
                imgpoints.append(sub_pixel_centers)
            
                img = cv2.drawChessboardCorners(img, chessboard_size, sub_pixel_centers, ret)
                camera.show('img',img)
            
                delete_image = os.path.basename(image)
                
        K = np.zeros((3,3))
        D = np.zeros((4,1))
        rvecs = []
        tvecs = []
            
        rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], K, D ,rvecs, tvecs, cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6))

        print('Used images num :',used_images_num)
        print("RMS:",rms)
        print("K:",K)
        print("D:",D)
    
        cv_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
        cv_file.write("K", K)
        cv_file.write("D",D)
        cv_file.release()

        return K, D, delete_image

    except Exception as e: 
        print(f'get_parameters_of_fisheye error: {e}')
        return None, None, delete_image

def test(folder_path):
    
    ret, frame = camera.get_image()

    if not ret:
        print("camera error")
        return None

    cv2.imwrite(f'{folder_path}/original_calibration.png',frame)

    cv_file = cv2.FileStorage(f'{folder_path}/calibration.yaml', cv2.FILE_STORAGE_READ)
    
    K = cv_file.getNode('K').mat()
    D = cv_file.getNode('D').mat()
    cv_file.release()

    h,w = frame.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), np.eye(3), balance=1.0)
    map1,map2 = cv2.fisheye.initUndistortRectifyMap(K,D, np.eye(3),new_K, (w,h), cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
       
    #x,y,w,h = roi
    #undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    cv2.imwrite(f'{folder_path}/undistorted_calibration.png',undistorted_frame)

    camera.release()
    
    trimmed_frame,x_ratio, y_ratio = detect_ar_marker(undistorted_frame)

    gray = cv2.cvtColor(trimmed_frame,cv2.COLOR_BGR2GRAY)
    
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,75, 2)
  
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
    
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(trimmed_frame,[max_contour],-1, (0,255,0),1)

        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #x_coords = [point[0][0] for point in max_contour]
        x_min = tuple(max_contour[max_contour[:,:,0].argmin()][0])
        x_max = tuple(max_contour[max_contour[:,:,0].argmax()][0])

        dist = x_ratio*(x_max[0] - x_min[0])
        print(f'distance : {dist} mm')

        cv2.line(trimmed_frame, (x_min[0],230), (x_max[0],230), (0,0,255), 5)        
        cv2.putText(trimmed_frame,"Length: {: .2f} mm".format(dist), (1000,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
       
        cv2.imwrite(f'{folder_path}/test_calibration.png',trimmed_frame)

        return dist,trimmed_frame
    
    else:
        print("No contours found")
        return None, None

    
def detect_ar_marker(frame):

    #img = cv2.imread(image)
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary,parameters=parameters)

    frame_width_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    camera.show('Detected ArUco Markers',frame_width_markers)
    
    m = np.empty((4,2))

    corners2 = [np.empty((1,4,2)) for _ in range(4)]

    #print(corners2)
    
    for i,c in zip(ids.ravel(), corners):
        corners2[i] = c.copy()

    m[0] = corners2[0][0][2]
    m[1] = corners2[1][0][3]
    m[2] = corners2[2][0][0]
    m[3] = corners2[3][0][1]
    
    marker_coordinates = np.float32(m)

    qr_code_x_min = int((m[1][0]+m[0][0])//2 - 200)
    qr_code_x_max = int((m[1][0]+m[0][0])//2 + 200)
    qr_code_y_min = int(corners2[0][0][1][1] - 200)
    qr_code_y_max = int(corners2[0][0][2][1] + 200)
    
    qr_code = utils.get_qr_code_data(frame[qr_code_y_min:qr_code_y_max,qr_code_x_min:qr_code_x_max])

    x_dis = int(qr_code.split(',')[0]) if qr_code else 200  
    y_dis = int(qr_code.split(',')[1]) if qr_code else 150

    width = int(np.linalg.norm(marker_coordinates[1] - marker_coordinates[0]))
    height = int(np.linalg.norm(marker_coordinates[3] - marker_coordinates[0]))

    x_ratio = x_dis / width;
    y_ratio = y_dis / height;
    
    true_coordinates = np.float32([[0,0], [width,0], [width,height], [0,height]])

    mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)
    
    frame_trans = cv2.warpPerspective(frame, mat, (width,height))

    return frame_trans, x_ratio, y_ratio
