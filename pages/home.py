import tkinter as tk
import lib.calibration as calibration
from tkinter import messagebox
from datetime import datetime

from lib.camera import Camera
import lib.utils as utils

class Home(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.controller = controller

        self.today = datetime.now().strftime('%Y-%m-%d')
        
        tk.Label(self, text="ホーム").pack(pady=20)
        tk.Button(self,text="自動キャリブレーションを開始する",command=self.auto_calibration).pack(pady=20)
        tk.Button(self,text="キャリブレーションの精度を確認する",command=self.test_calibration).pack(pady=20)
        tk.Button(self,text="測定を開始する",command=self.controller.show_measurement).pack(pady=20)
        tk.Button(self,text="胃の測定を開始する",command=self.controller.show_stomach_measurement).pack(pady=20)
        tk.Button(self,text="生殖器の測定を開始する",command=self.controller.show_genital_measurement).pack(pady=20)
        tk.Button(self,text="カメラを上下動させる",command=self.controller.show_move_camera).pack(pady=20)

    def auto_calibration(self):
        messagebox.showinfo("自動キャリブレーション",'キャリブレーションを開始します。')

        Camera().move_to_distance(35)
        
        folder_path = f'./data/{self.today}/calibration/images'
        utils.make_folder(folder_path)

        calibration.take_images(folder_path)
        calibration.get_parameters_of_fisheye((7,10),1,f'./data/{self.today}/calibration/calibration.yaml',folder_path)

        messagebox.showinfo("自動キャリブレーション",'キャリブレーションが完了しました。')

    def test_calibration(self):
        messagebox.showinfo("キャリブレーションの精度検証",'精度検証用シートを置いてください。')

        camera = Camera()
        camera.move_to_distance(20)

        status = camera.adjust_to_marker()
        
        if status == "camera error":
            messagebox.showinfo("カメラの接続エラー","カメラを再接続してください。再接続のあと、「OK」を押してください。")
            status = camera.adjust_to_marker()
                            

        today = datetime.now().strftime('%Y-%m-%d')

        full_length = calibration.test(f'./data/{today}/calibration/')

        print(f'full_length: {full_length}mm')

        messagebox.showinfo('キャリブレーションの精度検証',f'full_length: {full_length}mm')
        
