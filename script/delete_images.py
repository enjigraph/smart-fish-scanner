import glob
import os

def main():
    delete_images('./calibration_images/single/camera_0/')
    delete_images('./calibration_images/single/camera_1/')
    delete_images('./calibration_images/stereo/camera_0/')
    delete_images('./calibration_images/stereo/camera_1/')
    
def delete_images(folder):

    images = glob.glob(os.path.join(folder,'*.png'))

    for image in images:
        print(f'delete {image}')
        os.remove(image)

    return 0

if __name__ == "__main__":
    main()
