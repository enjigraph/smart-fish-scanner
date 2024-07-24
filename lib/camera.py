import cv2
import serial
import time
import sys
import threading
import numpy as np
from lib.sensor import Sensor

class Camera:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Camera, cls).__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, *args, **kwargs):
        self.ser = serial.Serial('/dev/ttyACM0', 115200)
        time.sleep(2)
        self.cap = None
        
        self.sensor = Sensor()

    def on(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
            #print(decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC)Z)Z)
            print(f'{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)},{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
        
    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
    def get_image(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                print('camera is not detected')
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
                #print(decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC)))
                print(f'{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)},{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')

            return self.cap.read()
        except Exception as e:
            print(f'[get_image][error] {e}')
            return None,None
        
    def move_stepper(self,steps):

        _, distance = self.sensor.get_data()

        if float(distance) > 38 and int(steps) < 0:
            print(f'Moved stepper reached to upper limit')
            return 0

        command = f'{steps}\n'
        self.ser.write(command.encode())
        print(f'Moved stepper motor {steps} steps')

        start_time = time.time()
        
        while time.time() - start_time < 3:
            if self.ser.in_waiting>0:
                try:
                    line = self.ser.readline().decode('utf-8').rstrip()

                    if "completed" in line:
                        print(line)
                        break
                    else:
                        print(line)
                except:
                    pass
            else:
                self.ser.reset_input_buffer()
       
        #self.ser.reset_input_buffer()

        return 0
            
    def adjust_to_marker(self):

        errorNum = 0

        while True:
            ret, frame = self.get_image()

            if not ret:
                print(f'camera error: {errorNum}')
                errorNum += 1

                if errorNum > 3:
                    return "camera error"
                
                continue

            #self.show('ArUco Markers',frame)

            shift = self.determine_movement(frame)

            if shift == "up":
                print("UP")
                self.move_stepper(-6000)
            elif shift == "down":
                print("DOWN")
                self.move_stepper(6000)
            elif shift == "center":
                print("CENTER")            
                self.move_to_distance(25)
            elif shift == "skip":
                print("SKIP")
            elif shift == "not_moving":
                print("FINISH")
                break
            
        return 0

    def move_to_distance(self,target_distance):

        _, current_distance = self.sensor.get_data()
        
        distance = float(target_distance) - float(current_distance)
    
        while abs(distance) > 3:

            if not current_distance:
                continue

            coefficient = min(int(abs(distance)),4) if abs(distance) > 5 else 1
            print(f'coefficient: {coefficient}')
            
            if distance > 0:
                print("UP")
                self.move_stepper(-6000*coefficient)
            
            if distance < 0:
                print("DOWN")
                self.move_stepper(6000*coefficient)

            _, current_distance = self.sensor.get_data()
        
            if current_distance:
                distance = float(target_distance) - float(current_distance)
                print(str(distance))

        return 0
        
    def determine_movement(self,image):

        try:

            if image is None:
                print("image is None")
                return "skip"
            
            height, width = image.shape[:2]

            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters()

            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary,parameters=parameters)
            
            if ids is None:
                print("id is None")
                return "center"
        
            if len(ids) < 4:
                print("len(ids) < 4")
                return "center"

            image_width_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

            self.show('Detected ArUco Markers',image_width_markers)

            m = np.empty((4,2))
        
            corners2 = [np.empty((1,4,2)) for _ in range(4)]
            
            for i,c in zip(ids.ravel(), corners):
                corners2[i] = c.copy()

            m[0] = corners2[0][0][0]
            m[1] = corners2[1][0][1]
            m[2] = corners2[2][0][2]
            m[3] = corners2[3][0][3]
        
            print(f'{m[0]}, {m[1]}, {m[2]}, {m[3]}')
            print(f'[{width*0.05},{height*0.05}], [{width*0.95},{height*0.05}], [{width*0.95},{height*0.95}], [{width*0.05},{height*0.95}]')

            if m[0][0] > width*0.1 and m[0][1] > height*0.1 and m[1][0] < width*0.9 and m[1][1] > height*0.1 and m[2][0] < width*0.9 and m[2][1] < height*0.9 and m[3][0] > width*0.1 and m[3][1] < height*0.9:
                return "down"

            return "not_moving"

        except:
            return "skip"

    def show(self,title,frame):        
        try:
            if frame is None:
                print("frame is None")
                return 0;
            
            cv2.namedWindow(title,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title,640,480)
            cv2.imshow(title,frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f'[show][error] {e}')

        return 0
        
