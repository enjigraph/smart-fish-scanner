import cv2
import os
import pandas as pd
import numpy as np
from lib.camera import Camera

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

    cv2.destroyAllWindows()
    
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
    self.show('Undistorted Image',undistorted_frame)

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
    camera.show('Undistorted Image',undistorted_frame)
    
    cv2.imwrite(f'{folder_path}/undistorted_image.png',undistorted_frame)
    print(f'{folder_path}/undistorted_image.png saved.')

    return undistorted_frame


def get_length(frame,folder_path):
    
    square_size_mm = 24
    
    trimmed_frame = detect_ar_marker(frame)

    cv2.imwrite(f'{folder_path}/trimmed_image.png',trimmed_frame)

    full_length, x_head = get_full_length(trimmed_frame.copy(),folder_path)
    
    head_and_scales_length = get_head_and_scales_length(trimmed_frame.copy(),x_head,folder_path)

    fork_length = get_fork_length(trimmed_frame.copy(),folder_path)
    
    return full_length, head_and_scales_length, fork_length


def get_full_length(frame,folder_path):

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 2)
    #edges = cv2.Canny(gray, 50, 150)
 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
    
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame,[max_contour],-1, (0,255,0),1)

        cv2.namedWindow('Max Contor',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Max Contor',800,600)
        cv2.imshow('Max Contor',frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #x_coords = [point[0][0] for point in max_contour]
        x_min = tuple(max_contour[max_contour[:,:,0].argmin()][0])#min(x_coords)
        x_max = tuple(max_contour[max_contour[:,:,0].argmax()][0])#max(x_coords)

        dist = x_max[0] - x_min[0]
        print(f'distance : {dist} mm')

        cv2.line(frame, (x_min[0],50), (x_max[0],50), (0,0,255), 1)        
        cv2.putText(frame,"Length: {: .2f} mm".format(dist), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
        cv2.namedWindow('Image with Head and Tail',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image with Head and Tail',800,600)

        cv2.imshow('Image with Head and Tail',frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        cv2.imwrite(f'{folder_path}/full_length.png',frame)

        return dist, x_min[0]
    
    else:
        print("No contours found")
        return None, None

def get_head_and_scales_length(frame,x_head,folder_path):
    return 0
        
def get_fork_length(frame,folder_path):
    return 0


def detect_ar_marker(frame):

    #img = cv2.imread(image)

    x_dis, y_dis, size = 200, 150, 1
    
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

    width, height = (x_dis*size, y_dis*size)
    x_ratio = width / x_dis;
    y_ratio = height / y_dis;
    
    marker_coordinates = np.float32(m)
    true_coordinates = np.float32([[0,0], [width,0], [width,height], [0,height]])

    mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)
    
    frame_trans = cv2.warpPerspective(frame, mat, (width,height))

    cv2.namedWindow('Transform Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Transform Image',800,600)
    cv2.imshow('Transform Image',frame_trans)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return frame_trans