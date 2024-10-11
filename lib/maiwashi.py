import sys
import cv2
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import defaultdict
import traceback

import lib.get_min_diff_x as get_min_diff_x

def get_pull_length(frame,filename, x_ratio, y_ratio):
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img,np.array([0,0,0]),np.array([180,255,120]))
    result = cv2.bitwise_and(frame,frame,mask=mask)
    gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    _, edges = cv2.threshold(gray,1, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
    
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame,[max_contour],-1, (0,255,0),1)

        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #x_coords = [point[0][0] for point in max_contour]
        x_min = tuple(max_contour[max_contour[:,:,0].argmin()][0])
        x_max = tuple(max_contour[max_contour[:,:,0].argmax()][0])

        min_y = np.min(max_contour[:,:,1])
        max_y = np.max(max_contour[:,:,1])

        dist = x_ratio * x_max[0] - 20
        print(f'distance : {dist} mm')

        cv2.line(frame, (x_min[0],230), (x_max[0],230), (0,0,255), 5)        
        cv2.putText(frame,"Length: {: .2f} mm".format(dist), (1000,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        cv2.imwrite(f'{folder_path}/full_length.png',frame)

        return dist, x_max[0], max_contour
    else:
        print("No contours found")
        return None, None, None

def get_thin_point(frame,contour,x_tail,folder_path):

    try:
        x_points = []
        for point in contour:
            if point[0][0] > int(x_tail*0.8):
                x_points.append(point[0][0])
                
        hist, bin_edges = np.histogram(x_points, bins=50)
            
        hist_mean = np.mean(hist)

        plt.clf()
        plt.bar(bin_edges[:-1], hist, width=10)
        plt.axhline(y=hist_mean, color='r',linestyle='--')
        plt.savefig(f'{folder_path}/hist.png')
        
        x_start = int(x_tail*0.75)
        x_end = get_x_end(hist,bin_edges,hist_mean)
        
        if not x_end:
            print('x_end is None')
            x_end = int(x_tail*0.9)
        else:
            x_end = int(x_end)
        
        x_peak = get_peaks(contour,x_start,x_end)
        print(f'{x_start} - {x_end}')
        print(f'x_peak: {x_peak}')
    
        thin_x = get_min_diff_x(frame,contour,x_start,x_end,folder_path)
        print(f'thin_x: {thin_x}')
    
        if x_peak is not None and abs(thin_x - x_peak) > 50 and abs(thin_x - x_end) < 20:
            print(f'thin_x - x_peak, {x}, {x_end}')
            thin_x = get_min_diff_x(frame,contour,x_start,x_peak+2,folder_path)
        
        cv2.putText(frame,f'{filename.split("/")[1]} {filename.split("/")[3]}', (200,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        cv2.putText(frame,f'thin point: {thin_point}', (1000,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        cv2.line(frame,(thin_point,0),(thin_point,frame.shape[0]),(0,0,255),2)
        cv2.imwrite(f'{folder_path}/thin_point.png',frame)
        
    except Exception as e:
        error_info = traceback.extract_tb(e.__traceback__)
        _, line_number, _, _ = error_info[-1]
        print(f'error: [{line_number}] {e}')
        return 0

def get_peaks(contour,x_start,x_end):
    
    y_coords_in_range = [point[0][1] for point in contour if x_start <= point[0][0] <= x_end]
      
    y_mid_point = int((min(y_coords_in_range) + max(y_coords_in_range)) // 2)
    
    back_contour = [point[0] for point in contour if point[0][0] in range(x_start,x_end) and point[0][1] < y_mid_point]
      
    x_coords = [pt[0] for pt in back_contour]
    y_coords = [pt[1] for pt in back_contour]
      
    sorted_indices = np.argsort(x_coords)
    x_coords_sorted = np.array(x_coords)[sorted_indices]
    y_coords_sorted = np.array(y_coords)[sorted_indices]
    
    thin_point = max(back_contour, key=lambda point: point[1])
    thin_point = thin_point[0]
        
    peaks, _ = find_peaks(y_coords_sorted) 
    if peaks.size > 0:
        max_peak_value = np.max(y_coords_sorted[peaks])
        max_peak_indices = np.where(y_coords_sorted == max_peak_value)[0]
        thin_point = np.max(x_coords_sorted[max_peak_indices])
        print(f'peak x: {thin_point}, {max_peak_value}')
        return thin_point
    
    print("指定したx領域内に極大値がありません。")
    return None
    
def get_x_end(hist,bin_edges,hist_mean):
    max_idx = np.argmax(hist) 
    max_x = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2 

    x_below_mean = None
    for i in range(max_idx, 0, -1):
        if hist[i] < hist_mean:
            x_below_mean = bin_edges[i-1]-50
            break

    print(f'x_below_mean: {x_below_mean}')
    return x_below_mean

