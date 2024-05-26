import cv2
import numpy as np
from scipy.spatial import distance as dist

def main():
    get_length('undistorted_image.png')

def get_length(image):
    square_size_mm = 24
    
    img = detect_ar_marker(image)
    
    x_head = get_full_length(img.copy())
    
    get_head_and_scales_length(img.copy(),x_head)
    
def detect_ar_marker(image):

    img = cv2.imread(image)

    x_dis, y_dis, size = 200, 150, 1
    
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary,parameters=parameters)

    img_width_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

    cv2.imshow('Detected ArUco Markers',img_width_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
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
    
    img_trans = cv2.warpPerspective(img, mat, (width,height))

    cv2.namedWindow('Transform Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Transform Image',800,600)
    cv2.imshow('Transform Image',img_trans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_trans

def get_full_length(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 2)
    #edges = cv2.Canny(gray, 50, 150)
 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
    
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img,[max_contour],-1, (0,255,0),1)

        cv2.namedWindow('Max Contor',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Max Contor',800,600)
        cv2.imshow('Max Contor',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #x_coords = [point[0][0] for point in max_contour]
        x_min = tuple(max_contour[max_contour[:,:,0].argmin()][0])#min(x_coords)
        x_max = tuple(max_contour[max_contour[:,:,0].argmax()][0])#max(x_coords)

        dist = x_max[0] - x_min[0]
        print(f'distance : {dist} mm')

        cv2.line(img, (x_min[0],50), (x_max[0],50), (0,0,255), 1)        
        cv2.putText(img,"Length: {: .2f} mm".format(dist), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
        cv2.namedWindow('Image with Head and Tail',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image with Head and Tail',800,600)
        cv2.imshow('Image with Head and Tail',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite('full_length.png',img)

        return x_min[0]
    
    else:
        print("No contours found")

def get_head_and_scales_length(img,x_head):

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,np.array([90,50,50]),np.array([130,255,255]))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
    
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img,[max_contour],-1, (0,255,0),1)

        cv2.namedWindow('Max Contor',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Max Contor',800,600)
        cv2.imshow('Max Contor',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        M = cv2.moments(max_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        #x_coords = [point[0][0] for point in max_contour]
        x_min = tuple(max_contour[max_contour[:,:,0].argmin()][0])#min(x_coords)
        x_max = tuple(max_contour[max_contour[:,:,0].argmax()][0])#max(x_coords)

        dist = x_max[0] - x_head
        print(f'distance : {dist} mm')

        cv2.line(img, (x_head,50), (x_max[0],50), (0,0,255), 1)        
        cv2.putText(img,"Length: {: .2f} mm".format(dist), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255),1)
        cv2.namedWindow('Image with Head and Tail',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image with Head and Tail',800,600)
        cv2.imshow('Image with Head and Tail',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imwrite('head_and_scales_length.png',img)
    
    else:
        print("No contours found")
    

if __name__ == "__main__":
    main()
