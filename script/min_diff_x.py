import traceback
import sys
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get(frame,contour,x_start,x_end,folder_path):

    try:

        y_coords_in_range = [point[0][1] for point in contour if x_start <= point[0][0] <= x_end]
    
        y_mid_point = int((min(y_coords_in_range) + max(y_coords_in_range)) // 2)
        
        min_back_contour = {}
        max_front_contour = {}
    
        for point in contour:
            x, y = point[0]
    
            if x in range(x_start, x_end) and y < y_mid_point:
                if x not in min_back_contour or y < min_back_contour[x]:
                    min_back_contour[x] = y
                
            if x in range(x_start, x_end) and y > y_mid_point:
                if x not in max_front_contour or y > max_front_contour[x]:
                    max_front_contour[x] = y

        back_contour = [(x, y) for x, y in min_back_contour.items()]
        front_contour = [(x, y) for x, y in max_front_contour.items()]
        back_contour = sorted(back_contour,key=lambda x:x[0])
  
        min_diff = float('inf')
        min_diff_x = None
        diff_values = []
        x_values = []
        
        front_contour_dict = {point[0]: point[1] for point in front_contour}

        for back_point in back_contour:
            x_back, y_back = back_point
            if x_back in front_contour_dict:
                y_front = front_contour_dict[x_back]
                diff = abs(y_back - y_front)
                x_values.append(x_back)
                diff_values.append(diff)
                if diff < min_diff:
                    min_diff = diff
                    min_diff_x = x_back
    
        for point in back_contour:
            cv2.circle(frame,tuple(point),0,(0,255,0),-1)

        for point in front_contour:
            cv2.circle(frame,tuple(point),0,(0,255,0),-1)

        plt.clf()
        back_contour = np.array(back_contour)
        front_contour = np.array(front_contour)
        plt.plot(back_contour[:, 0], back_contour[:, 1],linestyle='--',label='back',color=(0,1,0))
        plt.plot(front_contour[:, 0], front_contour[:, 1],linestyle='--',label='front',color=(0,1,0))
        plt.plot(x_values, diff_values, color='green', label='diff')
        plt.axvline(x=min_diff_x, color='r', linestyle='-', label=f'x = {min_diff_x}')
        plt.legend()
        plt.savefig(f'{folder_path}/diff.png')

        return min_diff_x

    except Exception as e:
        error_info = traceback.extract_tb(e.__traceback__)
        _, line_number, _, _ = error_info[-1]
        print(f'error: [{line_number}] {e}')
        return None
