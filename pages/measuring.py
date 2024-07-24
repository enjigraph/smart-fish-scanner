import threading
import time
import os
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import lib.measurement as measurement
import lib.utils as utils
import lib.voice as voice

from lib.camera import Camera
from lib.digital_scale import DigitalScale
from lib.sensor import Sensor


class Measuring(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.controller = controller

        self.status_label = tk.Label(self, text="測定開始の準備ができました。\nイワシを置いてから、赤外線センサーで開始のタイミングを指示してください。")
        self.status_label.pack(pady=20)
        
        self.measurement_date_label = tk.Label(self, text="")
        self.measurement_date_label.pack(pady=5)
        
        self.capture_date_label = tk.Label(self, text="")
        self.capture_date_label.pack(pady=5)
        
        self.capture_location_label = tk.Label(self, text="")
        self.capture_location_label.pack(pady=5)

        self.latitude_label = tk.Label(self, text="")
        self.latitude_label.pack(pady=5)

        self.longitude_label = tk.Label(self, text="")
        self.longitude_label.pack(pady=5)
        
        self.species_label = tk.Label(self, text="")
        self.species_label.pack(pady=5)

        self.finish_button = tk.Button(self,text="測定を終了する",command=self.stop)
        self.finish_button.pack(pady=5)
        self.return_button = tk.Button(self,text="戻る",command=self.reset)
        
        
    def on_page_open(self):
        self.measurement_date_label.config(text="測定日: "+self.controller.shared_data.get("measurement_date",""))
        self.capture_date_label.config(text="採集日: "+self.controller.shared_data.get("capture_date",""))
        self.capture_location_label.config(text="採集場所: "+self.controller.shared_data.get("capture_location",""))
        self.latitude_label.config(text="緯度: "+self.controller.shared_data.get("latitude",""))
        self.longitude_label.config(text="経度: "+self.controller.shared_data.get("longitude",""))
        self.species_label.config(text="種: "+self.controller.shared_data.get("species",""))

        self.today = datetime.now().strftime('%Y-%m-%d')
        utils.make_folder(f'./data/{self.today}/images')
        
        self.is_running = True
        self.lock = False
        threading.Thread(target=self.loop).start()

    def loop(self):
        count = sum(os.path.isdir(os.path.join(f'./data/{self.today}/images/',name)) for name in os.listdir(f'./data/{self.today}/images/'))
        camera = Camera()
        digital_scale = DigitalScale()
        sensor = Sensor()
       
        last_weight = 0
        camera.on()

        check_weight = 0
        check_full_length = 0

        while self.is_running:
            ir_value, distance = sensor.get_data()
            #print(f'IR Value: {ir_value}, Distance: {distance}, lock:{self.lock}')

            if ir_value == "0" and not self.lock:
                voice.start()

                if digital_scale.get_stable_data_length() < 2:
                    voice.retry()
                    self.lock = False
                    continue
                
                self.status_label.config(text=f'{count+1}つ目のデータを測定中')
                print(f'start to get data :{count}')
                self.lock = True

                folder_path = f'./data/{self.today}/images/{count}/full_body'
                utils.make_folder(folder_path)

                status = camera.adjust_to_marker()
  
                if status == "camera error":
                    messagebox.showinfo("カメラの接続エラー","カメラを再接続してください。再接続のあと、「OK」を押してください。")
                    status = camera.adjust_to_marker()
                            
                original_image = measurement.get_image(f'{folder_path}/original_image.png')
                undistorted_image = measurement.undistort_fisheye_image(original_image,f'./data/{self.today}/calibration/calibration.yaml',folder_path)
                full_length, head_and_scales_length, fork_length = measurement.get_length(undistorted_image,folder_path)
                print(f'full_length: {full_length}mm')

                weight = digital_scale.get_weight()
                print(f'weight: {weight}')

                data = {'count':[count],'species':[self.controller.shared_data.get("species","")], 'measurement_date':[self.controller.shared_data.get("measurement_date","")],'capture_date':[self.controller.shared_data.get("capture_date","")],'capture_location':[self.controller.shared_data.get("capture_location","")],'latitude':[self.controller.shared_data.get("latitude","")],'longitude':[self.controller.shared_data.get("longitude","")],'full_length':[full_length],'weight':[weight],'image':[folder_path]}
                measurement.save(f'./data/{self.today}/result.csv',data)

                if abs(check_full_length - full_length) < 2 and abs(check_weight - weight) < 2:
                    voice.data_alert()
                    messagebox.showinfo("計測データに関するアラート","全長と重さが非常に近いデータが保存されました。")
                    
                camera.move_to_distance(20)
                count += 1

                last_weight = weight
                self.lock = False
                self.status_label.config(text="測定開始の準備ができました。\n赤外線センサーで開始のタイミングを指示してください。")

                check_weight = weight
                check_full_length = full_length
                
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
        self.is_running = False
        self.status_label.config(text="測定を終了しました")
        self.finish_button.pack_forget()
        self.return_button.pack(pady=10)


    def reset(self):
        self.status_label.config(text="測定開始の準備ができました。\nイワシを置いてから、赤外線センサーで開始のタイミングを指示してください。")
        self.return_button.pack_forget()
        self.finish_button.pack(pady=10)
        
        self.controller.show_home()
