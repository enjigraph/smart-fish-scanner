import cv2
import numpy as np
import glob

def main():
    
    calibration((7,10),1,"calibration.yaml","./calibration_images/single/camera_0")

    return 0

def calibration(chessboard_size,square_size_mm,file_name,calibration_images_folder):

    objp = np.zeros(( chessboard_size[0]*chessboard_size[1], 3 ), np.float32 )
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2) * square_size_mm

    objpoints = []
    imgpoints = []

    images = glob.glob(f'{calibration_images_folder}/*.png')

    used_images_num = 0
    
    for frame in images:
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        ret, centers = cv2.findChessboardCorners(gray,chessboard_size,flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        print(f'{frame} : {ret}')
        
        if ret:
            used_images_num += 1
        
            objpoints.append(objp)

            sub_pixel_centers = cv2.cornerSubPix(gray, centers, (5,5), (-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            imgpoints.append(sub_pixel_centers)
            
            img = cv2.drawChessboardCorners(img, chessboard_size, sub_pixel_centers, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
            

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
        mean_error += error
        
    print("total error: {}".format(mean_error/len(objpoints)))

    cv_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", camera_matrix)
    cv_file.write("dist_coeffs",dist_coeffs)
    cv_file.release()

    cv2.destroyAllWindows()

    return camera_matrix, dist_coeffs

if __name__ == "__main__":
    main()
