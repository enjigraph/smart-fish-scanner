import cv2
import os
import pandas as pd
import numpy as np
import lib.utils as utils
import lib.maiwashi as maiwashi
import lib.katakuchiiwashi as katakuchiiwashi
import lib.katakuchiiwashi_small as katakuchiiwashi_small
from lib.camera import Camera
import traceback

camera = Camera()
    
def save(file_path,data):

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_new = pd.DataFrame(data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(file_path, index=False)
        print(f'{file_path}にデータを追加しました。')
    else:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        print(f'新規CSVファイルを{file_path}に作成し、データを追加しました。')

def add_genital_weight_to_file(file_path,count,data):

    try:
        df = pd.read_csv(file_path)

        df.at[count,'genital_weight'] = data['genital_weight']
    
        df.to_csv(file_path, index=False)
        print(f'{file_path}にデータを追加しました。')
    except Exception as err:
        print(err)

def add_stomach_weight_to_file(file_path,count,data):

    try:
        df = pd.read_csv(file_path)

        df.at[count,'stomach_weight'] = data['stomach_weight']
    
        df.to_csv(file_path, index=False)
        print(f'{file_path}にデータを追加しました。')
    except Exception as err:
        print(err)

def get_image(file_path):

    ret, frame = camera.get_image()

    if not ret:
        print("camera error")
        return None

    cv2.imwrite(file_path,frame,[cv2.IMWRITE_JPEG_QUALITY,100])
    
    return frame

def undistort_image(frame,calibration_file_path,folder_path):
    
    cv_file = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
    
    camera_matrix = cv_file.getNode('camera_matrix').mat()
    dist_coeffs = cv_file.getNode('dist_coeffs').mat()
    cv_file.release()

    h,w = frame.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted_frame = cv2.undistort(frame,camera_matrix,dist_coeffs, None,new_camera_mtx)
       
    x,y,w,h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    
    cv2.imwrite(f'{folder_path}/undistorted_image.png',undistorted_frame)
    print(f'{folder_path}/undistorted_image.png saved.')

    return undistorted_frame

def undistort_fisheye_image(frame,calibration_file_path,folder_path):
    
    cv_file = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
    
    K = cv_file.getNode('K').mat()
    D = cv_file.getNode('D').mat()
    cv_file.release()

    h,w = frame.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), np.eye(3), balance=1.0)
    map1,map2 = cv2.fisheye.initUndistortRectifyMap(K,D, np.eye(3),new_K, (w,h), cv2.CV_16SC2)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
       
    #x,y,w,h = roi
    #undistorted_frame = undistorted_frame[y:y+h, x:x+w]
    
    cv2.imwrite(f'{folder_path}/undistorted_image.png',undistorted_frame)
    print(f'{folder_path}/undistorted_image.png saved.')

    return undistorted_frame


def get_length(frame,species,folder_path):
        
    trimmed_frame, x_ratio, y_ratio = trim_ar_region(frame,folder_path)

    if "マイワシ" in species:

        full_length, x_tail, contour, full_length_frame = maiwashi.get_full_length(trimmed_frame.copy(),folder_path, x_ratio, y_ratio)

        thin_point_frame = maiwashi.get_thin_point(trimmed_frame.copy(),contour,x_tail,x_ratio,folder_path)
    
        return full_length, full_length_frame, thin_point_frame

    if "カタクチイワシ(小)" in species:
        full_length, x_tail, contour, full_length_frame = katakuchiiwashi_small.get_full_length(trimmed_frame.copy(),folder_path, x_ratio, y_ratio)

        thin_point_frame = katakuchiiwashi_small.get_thin_point(trimmed_frame.copy(),contour,x_tail,x_ratio,folder_path)
    
        return full_length, full_length_frame, thin_point_frame

    if "カタクチイワシ" in species:
        full_length, x_tail, contour, full_length_frame = katakuchiiwashi.get_full_length(trimmed_frame.copy(),folder_path, x_ratio, y_ratio)

        thin_point_frame = katakuchiiwashi.get_thin_point(trimmed_frame.copy(),contour,x_tail,x_ratio,folder_path)
    
        return full_length, full_length_frame, thin_point_frame

    return None, None, None
                    
def trim_ar_region(frame,folder_path):

    #img = cv2.imread(image)
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary,parameters=parameters)

    frame_width_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

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
    
    qr_code = utils.get_qr_code_data(frame[qr_code_y_min:qr_code_y_max,qr_code_x_min:qr_code_x_max],folder_path)

    x_dis = int(qr_code.split(',')[0]) if qr_code else 250  
    y_dis = int(qr_code.split(',')[1]) if qr_code else 150
 
    #width, height = (x_dis*size, y_dis*size)
    
    width = int(np.linalg.norm(marker_coordinates[1] - marker_coordinates[0]))
    height = int(np.linalg.norm(marker_coordinates[3] - marker_coordinates[0]))

    x_ratio = x_dis / width;
    y_ratio = y_dis / height;
    
    true_coordinates = np.float32([[0,0], [width,0], [width,height], [0,height]])

    mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)

    frame_trans = cv2.warpPerspective(frame, mat, (width,height))

    cv2.imwrite(f'{folder_path}/trimmed_image.png',frame_trans)

    return frame_trans, x_ratio, y_ratio
