import tkinter as tk
from datetime import datetime
import threading
import time

import lib.measurement as measurement
import lib.utils as utils
import lib.voice as voice

from lib.camera import Camera
from lib.digital_scale import DigitalScale
from lib.sensor import Sensor

class GenitalMeasurement(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.controller = controller

        self.today = datetime.now().strftime('%Y-%m-%d')
        
        self.is_running = False

        self.status_label = tk.Label(self, text="生殖腺の測定を行う")
        self.status_label.pack(pady=20)

        self.start_button = tk.Button(self,text="測定開始",command=self.start)
        self.start_button.pack(pady=10)

        self.finish_button = tk.Button(self,text="測定を終了する",command=self.stop)

        self.return_button = tk.Button(self,text="戻る",command=self.reset)
        self.return_button.pack(pady=30)

    def start(self):
        utils.make_folder(f'./data/{self.today}/images')
        self.status_label.config(text="測定開始の準備ができました。\n赤外線センサーで開始のタイミングを指示してください。")

        self.start_button.pack_forget()
        self.return_button.pack_forget()
        self.finish_button.pack(pady=10)
        
        self.is_running = True
        self.lock = False

        threading.Thread(target=self.loop).start()

    def loop(self):
        count = utils.count_column_elements(f'./data/{self.today}/result.csv','genital_weight') 
        sensor = Sensor()
        camera = Camera()
        digital_scale = DigitalScale()
        
        last_weight = 0
        camera.on()

        while self.is_running:
            ir_value, distance = sensor.get_data()
            #print(f'IR Value: {ir_value}, Distance: {distance}, lock: {self.lock}')

            if ir_value == "0" and not self.lock:
                voice.start()

                if digital_scale.get_stable_data_length() < 2:
                    voice.retry()
                    self.lock = False
                    continue

                self.status_label.config(text=f'{count+1}つ目のデータを測定中')
                print(f'start to get data: {count}')
                self.lock = True

                folder_path = f'./data/{self.today}/images/{count}/genital'
                utils.make_folder(folder_path)
                
                status = camera.adjust_to_marker()
                      
                if status == "camera error":
                    messagebox.showinfo("カメラの接続エラー","カメラを再接続してください。再接続のあと、「OK」を押してください。")
                    status = camera.adjust_to_marker()
                
                original_image = measurement.get_image(f'{folder_path}/original_image.png')
                undistorted_image = measurement.undistort_fisheye_image(original_image,f'./data/{self.today}/calibration/calibration.yaml',folder_path)
                measurement.trim_ar_marker(undistorted_image,folder_path)
                
                weight = digital_scale.get_weight()
                print(f'weight: {weight}')

                data = {'genital_weight':weight}
                measurement.add_genital_weight_to_file(f'./data/{self.today}/result.csv',count,data)
                
                camera.move_to_distance(20)
                count += 1

                time.sleep(1)

                last_weight = weight
                self.lock = False
                self.status_label.config(text="測定開始の準備ができました。\n赤外線センサーで開始のタイミングを指示してください。")

                voice.finish()

            else:
                now_weight = digital_scale.get_data()
                if count > 0 and now_weight and last_weight - now_weight > 1.0:
                    print(f'clear weight: {last_weight}')
                    digital_scale.clear_stable_data()
                    last_weight = 0
                    voice.remove()
                                        
                digital_scale.update_stable_data()
                time.sleep(0.1)

        camera.release()
      
    def stop(self):
        self.status_label.config(text="測定を終了しました")
        self.is_running = False

        self.finish_button.pack_forget()
        self.return_button.pack(pady=10)
       
    def reset(self):
        self.status_label.config(text="生殖腺の測定を行う")
        self.start_button.pack_forget()
        self.return_button.pack_forget()
        self.start_button.pack(pady=10)
        self.return_button.pack(pady=10)

        self.finish_button.pack_forget()
        
        self.controller.show_home()

