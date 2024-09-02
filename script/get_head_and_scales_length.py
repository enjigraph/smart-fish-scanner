import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
                    
def get_full_length(frame, x_ratio, y_ratio):

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
        cv2.imwrite(f'./full_length.png',frame)

        return dist, x_min[0], x_max[0], frame
    
    else:
        print("No contours found")
        return None, None, None

def is_point_in_contour(p,contour):
    p_float = (float(p[0]),float(p[1]))
    return cv2.pointPolygonTest(contour,p_float, False) > 0


def trim_ar_region(frame):

    #img = cv2.imread(image)
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary,parameters=parameters)

    frame_width_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)

    m = np.empty((4,2))

    corners2 = [np.empty((1,4,2)) for _ in range(4)]

    print(corners2)
    
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
    
    x_dis = 250  
    y_dis = 100
 
    #width, height = (x_dis*size, y_dis*size)
    
    width = int(np.linalg.norm(marker_coordinates[1] - marker_coordinates[0]))
    height = int(np.linalg.norm(marker_coordinates[3] - marker_coordinates[0]))

    x_ratio = x_dis / width;
    y_ratio = y_dis / height;
    
    true_coordinates = np.float32([[0,0], [width,0], [width,height], [0,height]])

    mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)

    frame_trans = cv2.warpPerspective(frame, mat, (width,height))

    cv2.imwrite(f'./trimmed_image.png',frame_trans)
    print(x_ratio)
    return frame_trans, x_ratio, y_ratio

try:

    frame = cv2.imread('./undistorted_image.png')
    trimmed_frame, x_ratio, y_ratio = trim_ar_region(frame)
    full_length, x_head, x_tail, full_length_frame = get_full_length(trimmed_frame.copy(), x_ratio, y_ratio)
    
    print(x_head)
    print(x_tail)
    hsv_img = cv2.cvtColor(trimmed_frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img,np.array([0,0,0]),np.array([180,255,35]))

    result = cv2.bitwise_and(trimmed_frame,trimmed_frame,mask=mask)
    gray_fins = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_fins[:,int(x_tail*0.8):x_tail], 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    external_contours = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
    
        #if x > int(x_tail*0.6) and x < x_tail:
        external_contours.append(contour)
    max_contour = max(contours, key=cv2.contourArea)

    points = np.array([point[0] for point in max_contour])
    
    x_coords = []
    y_coords = []

    for point in points:        
        base_x, base_y = point
        right_point = (base_x +1, base_y)
        #print(right_point)
        
        if is_point_in_contour(right_point, max_contour):
            cv2.circle(trimmed_frame[:,int(x_tail*0.8):x_tail],tuple(point),1,(255,0,0),-1)
            x_coords.append(base_x)
            y_coords.append(base_y)

    hist, bin_edges = np.histogram(x_coords, bins=50)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    
    peaks, _ = find_peaks(hist, distance=1)
    top_peaks = sorted(peaks, key=lambda x:hist[x], reverse=True)[:1]
    #print(top_peaks)
    
    x_ranges = [(bin_edges[peak],bin_edges[peak+1]) for peak in top_peaks]
    #print(x_ranges)
    vertical_bar_points = []

    for x_start, x_end in x_ranges:
        points_in_range = [point for point in max_contour if x_start <= point[0][0] < x_end]
        vertical_bar_points.extend(points_in_range)

    points_list = [tuple(point[0]) for point in vertical_bar_points]

    x_min = None
    if points_list:
        min_x_point = min(points_list, key=lambda p :p[0])
        max_x_point = max(points_list, key=lambda p :p[0])

        print(min_x_point)
        print(max_x_point)

        x_min = int(x_tail)*0.8+min_x_point[0]
        print(x_ratio * (int(x_tail)*0.8+min_x_point[0]-x_head))
        print(x_ratio * (int(x_tail)*0.8+max_x_point[0]-x_head))

        original_point = (int(x_min),min_x_point[1])
        cv2.circle(trimmed_frame[:,int(x_tail*0.8):x_tail],min_x_point,1,(0,0,255),-1)
        #cv2.circle(trimmed_frame[:,int(x_tail*0.8):x_tail],max_x_point,1,(0,0,255),-1)
        cv2.circle(trimmed_frame,original_point,1,(0,0,255),-1)


    vetical_bar_contour = np.array(vertical_bar_points)

    
    cv2.drawContours(trimmed_frame[:,int(x_tail*0.8):x_tail],[vetical_bar_contour], -1, (0,255,0),1)       
    cv2.imwrite(f'./head_and_scales_0.length.png',trimmed_frame[:,int(x_tail*0.8):x_tail])
        
    dist = x_ratio * (x_min - x_head)
        
    cv2.line(trimmed_frame, (x_head,250), (int(x_min),250), (0,0,255), 1)        
    cv2.putText(trimmed_frame,"Length: {: .2f} mm".format(dist), (1000,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        
    cv2.imwrite(f'./head_and_scales.length.png',trimmed_frame)
    
except Exception as e:
    print(f'get_head_and_scales_length error: {e}')
        
