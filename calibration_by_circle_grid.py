import cv2
import numpy as np
import glob
 
pattern_size = (4,11)
circle_spacing = 20

objp = np.zeros(( pattern_size[0]*pattern_size[1], 3 ), np.float32 )
objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
objp *= circle_spacing

objpoints = []
imgpoints = []

images = glob.glob('./calibration_images/*.png')

used_images_num = 0

for frame in images:
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, centers = cv2.findCirclesGrid(gray,pattern_size,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    print(f'{frame} : {used_images_num} : {ret}')
    if ret:
        used_images_num += 1
        
        objpoints.append(objp)
        imgpoints.append(centers)

        img = cv2.drawChessboardCorners(img, pattern_size, centers, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1], None,None)

print('Used images num :',used_images_num)
print('Image shape :',gray.shape[::-1])
print('RMS :',ret)
print("カメラ行列：\n",camera_matrix)
print("歪み係数：\n",dist_coeffs)

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    print(f'{i} : {error}')
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))

calibration_file = "calibration.yaml"

cv_file = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_matrix", camera_matrix)
cv_file.write("dist_coeffs",dist_coeffs)
cv_file.release()

