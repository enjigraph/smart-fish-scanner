import cv2

def main():
    take(0,'./calibration_images/camera_0')

def take(camera_id,folder_name):

    try:
        print(f'start to take a image camera_{camera_id}')

        cap = cv2.VideoCapture(camera_id)

        image_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            cv2.imshow('Frame', frame)

            key = cv2.waitKey(1)

            if key == ord('s'):
                cv2.imwrite(f'{folder_name}/image_{image_count}.png',frame)
                print(f'Image {image_count} saved.')
                image_count += 1
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except:
        print(f'error: camera_{camera_id} is not found')
        
        
    return 0

if __name__ == "__main__":
    main()
