import cv2
import numpy as np

calibration_file = 'calibration.yaml'

cv_file = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)

camera_matrix = cv_file.getNode('camera_matrix').mat()
dist_coeffs = cv_file.getNode('dist_coeffs').mat()
cv_file.release()

print("Camera matrix:\n",camera_matrix)
print("Distortion coefficients:\n",dist_coeffs)

cap = cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 20)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

while True:
    ret, frame = cap.read()

    if not ret:
        print("camera error")
        break

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break    
    elif key == ord('s'):
        cv2.imwrite(f'./target.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,100])
        print(f'Image captured.')

        h,w = frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        undistorted_frame = cv2.undistort(frame,camera_matrix,dist_coeffs, None,new_camera_mtx)

        #cv2.imwrite('undistorted_image_1.png',undistorted_frame)
       
        x,y,w,h = roi
        undistorted_frame = undistorted_frame[y:y+h, x:x+w]
        cv2.imshow('Undistorted Image',undistorted_frame)

        cv2.imwrite('undistorted_image.png',undistorted_frame)
        print(f'Image saved.')
        
        break
