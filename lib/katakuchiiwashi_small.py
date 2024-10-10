import sys
import cv2
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import defaultdict
import traceback

import lib.min_diff_x as min_diff_x

def get_full_length(frame,folder_path, x_ratio, y_ratio):

    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    print(hsv_img[420,1100])

    mask = cv2.inRange(hsv_img,np.array([0,0,0]),np.array([180,255,160]))
  
    result = cv2.bitwise_and(frame,frame,mask=mask)
    
    gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    _, edges = cv2.threshold(gray,1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    center_contours = []
    
    height, width = frame.shape[:2]
    center_x = width // 2

    # 各外形についてループ
    for contour in contours:
        # 外形の全点が中央ライン付近にあるか確認
        for point in contour:
            px, py = point[0]
        
            # x座標が中央ライン付近にある場合
            if abs(px - center_x) < 100:  # 調整可能なライン幅（±5ピクセル）
                center_contours.append(contour)
                break  # 一度でも中央を通る点があれば次の外形に移動

    print(len(center_contours))
            
    if center_contours:
    
        max_contour = max(center_contours, key=cv2.contourArea)
        cv2.drawContours(frame,[max_contour],-1, (0,255,0),1)

        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #x_coords = [point[0][0] for point in max_contour]
        x_min = tuple(max_contour[max_contour[:,:,0].argmin()][0])
        x_max = tuple(max_contour[max_contour[:,:,0].argmax()][0])
        
        dist = x_ratio * x_max[0] - 20
        print(f'distance : {dist} mm')

        cv2.line(frame, (x_min[0],230), (x_max[0],230), (0,0,255), 5)        
        cv2.putText(frame,"Length: {: .2f} mm".format(dist), (1000,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        cv2.imwrite(f'{folder_path}/full_length.png',frame)

        return dist, x_max[0], max_contour
    else:
        print("No contours found")
        return None, None, None

def get_thin_point(frame,contour,x_tail,x_ratio,folder_path):

    try:
        hist_mean = np.mean(hist)
        
        plt.clf()
        plt.bar(bin_edges[:-1], hist, width=10)
        plt.axhline(y=hist_mean, color='r',linestyle='--')
        plt.savefig(f'{folder_path}/hist.png')
        
        x_start = int(x_tail*0.86)
        #x_end = get_x_end(hist,bin_edges,hist_mean)
        x_end = int(x_tail*0.94)

        thin_point = min_diff_x.get(frame,contour,x_start,x_end,folder_path)

        dist = x_ratio * thin_x - 20
        cv2.putText(frame,"thin x: {: .2f} mm".format(dist), (1000,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),5)
        cv2.circle(frame,(830,420),2,(0,0,255),-1)
        cv2.line(frame,(thin_point,0),(thin_point,frame.shape[0]),(0,0,255),1)
        cv2.imwrite(f'{folder_path}/thin_point.png',frame)

    except Exception as e:
        error_info = traceback.extract_tb(e.__traceback__)
        _, line_number, _, _ = error_info[-1]
        print(f'error: [{line_number}] {e}')
