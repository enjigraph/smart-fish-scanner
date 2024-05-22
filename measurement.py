import cv2
import numpy as np
from scipy.spatial import distance as dist

square_size_mm = 24

img = cv2.imread('undistorted_image.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 2)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:

    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(img,[max_contour],-1, (0,255,0),2)

    cv2.imshow('Max Contor',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(img,[box],0, (255,0,0),2)
    
    cv2.imshow('Box',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    (tl, tr, br, bl) = box
    d1 = dist.euclidean(tl,tr)
    d2 = dist.euclidean(tr,br)

    length = max(d1, d2)
    cv2.putText(img,"Length: {: .2f} mm".format(length), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
    
    cv2.imshow('Length',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No contours found")
