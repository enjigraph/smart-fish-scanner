def get_head_and_scales_length(x_head,x_tail,frame,folder_path,x_ratio,y_ratio):

    try:

        hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img,np.array([0,0,0]),np.array([180,255,35]))

        result = cv2.bitwise_and(frame,frame,mask=mask)
        gray_fins = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray_fins[:,int(x_tail*0.8):x_tail], 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            points = np.array([point[0] for point in max_contour])
    
            x_coords = []
            y_coords = []

            for point in points:        
                base_x, base_y = point
                right_point = (base_x +1, base_y)
               
                if is_point_in_contour(right_point, max_contour):
                    cv2.circle(frame[:,int(x_tail*0.8):x_tail],tuple(point),1,(255,0,0),-1)
                    x_coords.append(base_x)
                    y_coords.append(base_y)

            hist, bin_edges = np.histogram(x_coords, bins=50)
            bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    
            peaks, _ = find_peaks(hist, distance=1)
            top_peaks = sorted(peaks, key=lambda x:hist[x], reverse=True)[:1]
            
            x_ranges = [(bin_edges[peak],bin_edges[peak+1]) for peak in top_peaks]
            vertical_bar_points = []

            for x_start, x_end in x_ranges:
                points_in_range = [point for point in max_contour if x_start <= point[0][0] < x_end]
                vertical_bar_points.extend(points_in_range)
        
            points_list = [tuple(point[0]) for point in vertical_bar_points]

            x_min = None
            if points_list:
                min_x_point = min(points_list, key=lambda p :p[0])
                max_x_point = max(points_list, key=lambda p :p[0])
                x_min = int(x_tail)*0.8+min_x_point[0]

                #print(min_x_point)
                #print(max_x_point)
                #print(x_ratio * (int(x_tail)*0.8+min_x_point[0]-x_head))
                #print(x_ratio * (int(x_tail)*0.8+max_x_point[0]-x_head))
                
                original_point = (int(x_min),min_x_point[1])
                cv2.circle(frame[:,int(x_tail*0.8):x_tail],min_x_point,1,(0,0,255),-1)
                #cv2.circle(frame[:,int(x_tail*0.8):x_tail],max_x_point,1,(0,0,255),-1)
                cv2.circle(frame,original_point,1,(0,0,255),-1)


            vetical_bar_contour = np.array(vertical_bar_points)
    
            cv2.drawContours(frame[:,int(x_tail*0.8):x_tail],[vetical_bar_contour], -1, (0,255,0),1)       
            cv2.imwrite(f'{folder_path}/head_and_scales_detail.png',frame[:,int(x_tail*0.8):x_tail])
        
            return 0

        else:
            print("No contours found")
            return None, None

    except Exception as e:
        print(f'get_head_and_scales_length error: {e}')
        return None, None
        
def is_point_in_contour(p,contour):
    p_float = (float(p[0]),float(p[1]))
    return cv2.pointPolygonTest(contour,p_float, False) > 0
