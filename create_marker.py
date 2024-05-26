import cv2
import numpy as np

def main():
    create_aruco_marker(0, 200)
    create_aruco_marker(1, 200)
    create_aruco_marker(2, 200)
    create_aruco_marker(3, 200)

def create_aruco_marker(marker_id, marker_size):

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker_image = cv2.aruco.generateImageMarker(dictionary,marker_id,marker_size)

    cv2.imwrite(F'marker_{marker_id}.png',marker_image)       
    
if __name__ == "__main__":
    main()

