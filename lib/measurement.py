import cv2
import os
import pandas as pd
import numpy as np
import lib.utils as utils
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


def get_length(frame,folder_path):
        
    trimmed_frame, x_ratio, y_ratio = trim_ar_region(frame,folder_path)

    full_length, x_head, x_tail, full_length_frame = get_full_length(trimmed_frame.copy(),folder_path, x_ratio, y_ratio)
    
    head_and_scales_length, head_and_scales_length_frame = get_head_and_scales_length(x_head,x_tail,trimmed_frame.copy(),folder_path, x_ratio, y_ratio)

    head_and_fork_length, head_and_fork_length_frame = get_head_and_fork_length(x_head,x_tail,trimmed_frame.copy(),folder_path, x_ratio, y_ratio)
    
    return full_length, head_and_scales_length, head_and_fork_length, full_length_frame, head_and_scales_length_frame, head_and_fork_length_frame


def get_full_length(frame, folder_path, x_ratio, y_ratio):

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,75, 2)
 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
    
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame,[max_contour],-1, (0,255,0),1)

        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #x_coords = [point[0][0] for point in max_contour]
        x_min = tuple(max_contour[max_contour[:,:,0].argmin()][0])
        x_max = tuple(max_contour[max_contour[:,:,0].argmax()][0])

        dist = x_ratio * (x_max[0] - x_min[0])
        print(f'distance : {dist} mm')

        cv2.line(frame, (x_min[0],230), (x_max[0],230), (0,0,255), 5)        
        cv2.putText(frame,"Length: {: .2f} mm".format(dist), (1000,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        cv2.imwrite(f'{folder_path}/full_length.png',frame)

        return dist, x_min[0], x_max[0], frame
    
    else:
        print("No contours found")
        return None, None, None

def get_head_and_scales_length(x_head,x_tail,frame,folder_path,x_ratio,y_ratio):

    try:
        hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img,np.array([0,0,0]),np.array([180,255,35]))

        result = cv2.bitwise_and(frame,frame,mask=mask)

        gray_fins = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray_fins, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        external_contours = []

        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
    
            if x > x_tail*0.85 and x < x_tail:
                external_contours.append(contour)

        sorted_external_contours = sorted(external_contours, key=calculate_change_rates,reverse=True)
        significant_points = find_significant_points([sorted_external_contours[0]])

        x_min = None

        for point in significant_points:
    
            color = hsv_img[point[1],point[0]-1]
            color2 = hsv_img[point[1],point[0]+1]
            if np.all(np.array([0,0,30]) < color) and  np.all(color < np.array([180,150,255])) and np.all(np.array([0,0,0]) < color2) and np.all(color2 < np.array([180,255,35])) :
                cv2.circle(frame,point,1,(0,255,0),-1)

                if not x_min or x_min > point[0]:
                    x_min = point[0]

        if not x_min:
            print('get_head_and_scales_length error: x_min is None')
            return None, None
        
        dist = x_ratio * (x_min - x_head)
        
        cv2.drawContours(frame,external_contours, -1, (0,255,0),1)
        cv2.line(frame, (x_head,50), (x_min,50), (0,0,255), 1)        
        cv2.putText(frame,"Length: {: .2f} mm".format(dist), (1000,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        
        cv2.imwrite(f'{folder_path}/head_and_scales.length.png',frame)
    
        return dist, frame

    except Exception as e:
        print(f'get_head_and_scales_length error: {e}')
        return None, None
        
def get_head_and_fork_length(x_head,x_tail,frame,folder_path,x_ratio,y_ratio):

    try:
    
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        blurred = cv2.GaussianBlur(gray, (5,5),0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:

            external_contours = []
           
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)

                #if x > x_tail*0.8 and x < x_tail and max(calculate_change_rates(contour)) > 1.5:
                if x > x_tail*0.9 and x < x_tail:
                    h = hierarchy[0][i]
                
                    if h[3] == -1 and get_angle_count(contour) > 1:
                        external_contours.append(contour)

                        #for point in contour:
                        #    if not x_min or point[0][0] < x_min:
                        #        x_min = point[0][0]

            sorted_external_contours = sorted(external_contours, key=calculate_change_rates,reverse=True)

            x_min = None
            for point in sorted_external_contours[0]:
                if not x_min or point[0][0] < x_min:
                    x_min = point[0][0]

            if not x_min:
                print('get_head_and_fork_length error: x_min is None')
                return None, None
        
            dist = x_ratio * (x_min - x_head)
            print(f'head and fork distance : {dist} mm')

            cv2.drawContours(frame,external_contours, -1, (0,255,0),1)
            cv2.line(frame, (x_head,50), (x_min,50), (0,0,255), 1)        
            cv2.putText(frame,"Length: {: .2f} mm".format(dist), (1000,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
            
            cv2.imwrite(f'{folder_path}/head_and_fork.length.png',frame)

            return dist, frame
    
        else:
            print("No contours found")
            return None, None
    except Exception as e:
        print(f'get_head_and_fork_length error: {e}')
        return None, None

def calculate_max_distance_from_point(point,contour):
    max_distance = 0
    farthest_point = point
    for p in contour:
        distance =np.linalg.norm(np.array(point) - np.array(p[0]))
        if distance > max_distance:
            max_distance = distance
            farthest_point = p[0]
    return farthest_point, max_distance

def calculate_change_rates(contour):
    change_rates = []
    num_points = len(contour)
    for i in range(num_points):
        curr_point = contour[i][0]

        farthest_point, max_distance = calculate_max_distance_from_point(curr_point,contour)
        
        dx = abs(curr_point[0] - farthest_point[0])
        dy = abs(curr_point[1] - farthest_point[1])

        if dx != 0:
            change_rate = dy / dx
        else:
            change_rate = 0
            
        change_rates.append(change_rate)

    return change_rates

def get_angle_count(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    count = 0
    for i in range(len(approx)):
        try:
            prev_point = approx[i-1][0]
            curr_point = approx[i][0]
            next_point = approx[(i+1) % len(approx)][0]
        
            vec1 = np.array(curr_point) - np.array(prev_point)
            vec2 = np.array(next_point) - np.array(curr_point)

            dot_product = np.dot(vec1,vec2)
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            
            if not np.isfinite(magnitude1) or not np.isfinite(magnitude2) or magnitude1 == 0 or magnitude2 == 0:
                continue
            
            angle = np.arccos(dot_product / (magnitude1*magnitude2))* 180 / np.pi

            if angle > 150:
                count += 1

            dx1 = curr_point[0] - prev_point[0]
            dy1 = curr_point[1] - prev_point[1]
            dx2 = next_point[0] - curr_point[0]
            dy2 = next_point[1] - curr_point[1]
        except:
            continue
            
    return count

def find_significant_points(contours):
    significant_points = []

    for contour in contours:
        for i in range(len(contour) - 1):
            x1,y1 = contour[i][0]
            x2,y2 = contour[i+1][0]
            dx = abs(x2-x1)
            dy = abs(y2-y1)
            #print(f'{x1},{y1}. {x2},{y2}')
            
            if dx>dy:
                significant_points.append((x1,y1))
    return significant_points
                    
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

#def save_trimmed_image(folder_path,frame,marker_coordinates):
#
#    width = int(np.linalg.norm(marker_coordinates[1] - marker_coordinates[0]))
#    height = int(np.linalg.norm(marker_coordinates[3] - marker_coordinates[0]))
#    
#    mat = cv2.getPerspectiveTransform(marker_coordinates, np.float32([[0,0], [width,0], [width,height], [0,height]]))#
#
#    cv2.imwrite(f'{folder_path}/trimmed_image.png',cv2.warpPerspective(frame, mat, (width,height)))
