import cv2
import time
import tkinter as tk
import os
import shutil
from tkinter import messagebox
from datetime import datetime
from PIL import Image,ImageTk

from lib.camera import Camera
import lib.calibration as calibration
import lib.utils as utils

camera = Camera()

class Home(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.controller = controller

        self.today = datetime.now().strftime('%Y-%m-%d')
        self.running = True
        tk.Label(self, text="ホーム").pack(pady=20)

        self.canvas = tk.Canvas(self,bg="lightgray",width=480,height=280)
        self.canvas.pack(pady=20)

        tk.Button(self,text="カメラを上下動させる",command=self.controller.show_move_camera).pack(pady=(5,20)) 
        tk.Button(self,text="手動キャリブレーションを開始する",command=self.manual_calibration).pack(pady=5)
        tk.Button(self,text="自動キャリブレーションを開始する",command=self.auto_calibration).pack(pady=5)
        tk.Button(self,text="過去のキャリブレーションファイルをコピーする",command=self.copy_calibration).pack(pady=5)
        tk.Button(self,text="キャリブレーションの精度を確認する",command=self.test_calibration).pack(pady=(5,20))
        tk.Button(self,text="測定を開始する",command=self.controller.show_measurement).pack(pady=5)
        tk.Button(self,text="胃の測定を開始する",command=self.controller.show_stomach_measurement).pack(pady=5)
        tk.Button(self,text="生殖腺の測定を開始する",command=self.controller.show_genital_measurement).pack(pady=5)

    def on_page_open(self):
        self.show_camera()
        
    def show_camera(self):

        if self.running:

            camera.on()

            ret, frame = camera.get_image()
            
            if not ret:
                messagebox.showinfo("カメラエラー",'エラーが発生しました。')

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame =self.resize_image(frame,self.canvas.winfo_width(),self.canvas.winfo_height())
            image = Image.fromarray(frame)
        
            self.imageTk = ImageTk.PhotoImage(image=image)        
            self.canvas.create_image(0,0,anchor=tk.NW, image=self.imageTk)
            
        self.after(500,self.show_camera)

    def resize_image(self,image,max_width,max_height):
        height, width, _ = image.shape
        aspect_ratio = width / height
        if width > max_width or height > max_height:
            if aspect_ratio > 1:
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            resize_image = cv2.resize(image, (new_width, new_height))

            return resize_image
        return image

    def copy_calibration(self):
        folders = [f for f in os.listdir('./data') if os.path.isdir(os.path.join('./data',f))]

        sorted_folders = sorted(folders, key=lambda x: datetime.strptime(x, '%Y-%m-%d'),reverse=True)

        utils.make_folder(f'./data/{self.today}/calibration/')

        for folder in sorted_folders:
            if os.path.isfile(f'./data/{folder}/calibration/calibration.yaml'):
                shutil.copy(f'./data/{folder}/calibration/calibration.yaml',f'./data/{self.today}/calibration/')
                messagebox.showinfo("キャリブレーションファイルのコピー",f'{folder}のキャリブレーションファイルをコピーしました。')
                break

    def manual_calibration(self):
        self.running = False
        
        response = messagebox.askquestion("手動キャリブレーション",'キャリブレーションを開始しますか？')

        if response == 'yes':

            folder_path = f'./data/{self.today}/calibration/images'
            utils.make_folder(folder_path)

            calibration.take_images_by_manual(folder_path)
            calibration.get_parameters_of_fisheye((7,10),1,f'./data/{self.today}/calibration/calibration.yaml',folder_path)

        else:
            pass
        
        self.running = True

    def auto_calibration(self):

        self.running = False
        
        response = messagebox.askquestion("自動キャリブレーション",'キャリブレーションを開始しますか？')

        if response == 'yes':

            start_time = time.time()
            folder_path = f'./data/{self.today}/calibration/images'
            utils.make_folder(folder_path)

            camera.move_to_distance(35)
           
            calibration.take_images(folder_path,['stop','forward','backward','backward','forward'],False)

            camera.move_stepper(-6000*4)
            
            calibration.take_images(folder_path,['stop','forward','right','backward','backward','left','forward'],True)
            calibration.get_parameters_of_fisheye((7,10),1,f'./data/{self.today}/calibration/calibration.yaml',folder_path)

            processing_time = time.time() - start_time
            print(f'processing_time: {processing_time}s')
            messagebox.showinfo("自動キャリブレーション",'キャリブレーションが完了しました。')

        else:
            pass

        self.running = True

    def test_calibration(self):
        response = messagebox.askquestion("キャリブレーションの精度検証",'精度検証を開始しますか？\n精度検証用シートを忘れずに置いてください。')

        if response == 'yes':

            self.running = False
            
            camera.on()
            camera.move_to_distance(20)

            status = camera.adjust_to_marker()
            
            if status == "camera error":
                messagebox.showinfo("カメラの接続エラー","カメラを再接続してください。再接続のあと、「OK」を押してください。")
                status = self.controller.camera.adjust_to_marker()

            camera.release()

            today = datetime.now().strftime('%Y-%m-%d')

            full_length, trimmed_frame = calibration.test(f'./data/{today}/calibration/')
            
            print(f'full_length: {full_length}mm')

            self.show_popup('キャリブレーションの精度検証',f'full_length: {full_length}mm',trimmed_frame)

        else:
            pass

        self.running = True
        
    def show_popup(self,title,text,frame):
        
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("800x480")

        tk.Label(popup, text=text).pack(pady=20)

        canvas = tk.Canvas(popup,bg="lightgray",width=480,height=280)
        canvas.pack(pady=20)

        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame = self.resize_image(frame,480,240)
        image = Image.fromarray(frame)
        
        self.popup_imageTk = ImageTk.PhotoImage(image=image)        
        canvas.create_image(0,0,anchor=tk.NW, image=self.popup_imageTk)

        self.after(3000, lambda: popup.destroy())
