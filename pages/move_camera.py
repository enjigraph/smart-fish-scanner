import tkinter as tk
from lib.camera import Camera

class MoveCamera(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.controller = controller
        self.camera = Camera()
        
        tk.Label(self, text="カメラを移動").pack(pady=20)
        tk.Button(self,text="カメラを高さ30cmへ移動させる",command=self.move_camera_to_center).pack(pady=20)
        tk.Button(self,text="カメラを上部に最大限移動させる",command=self.move_camera_to_upper_limit).pack(pady=20)
        tk.Button(self,text="カメラを上昇させる",command=self.up_camera).pack(pady=20)
        tk.Button(self,text="カメラを下降させる",command=self.down_camera).pack(pady=10)
        tk.Button(self,text="戻る",command=self.controller.show_home).pack(pady=30)

    def move_camera_to_center(self):
        self.camera.move_to_distance(30)

    def move_camera_to_upper_limit(self):
        self.camera.move_to_distance(35)
        
    def up_camera(self):
        self.camera.move_stepper(-6000)
        
    def down_camera(self):
        self.camera.move_stepper(6000)

