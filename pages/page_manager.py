import tkinter as tk

from pages.home import Home
from pages.preparing_for_measurement import PreparingForMeasurement
from pages.measurement import Measurement
from pages.measuring import Measuring
from pages.genital_measurement import GenitalMeasurement
from pages.stomach_measurement import StomachMeasurement
from pages.move_camera import MoveCamera

class PageManager(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Smart Fish Scanner')
        
        self.geometry("600x800")
        self.resizable(False,False)

        self.contanier = tk.Frame(self)
        self.contanier.pack(side="top", fill="both", expand=True)
        self.contanier.grid_rowconfigure(0,weight=1)
        self.contanier.grid_columnconfigure(0,weight=1)
                
        self.frames = {}
        self.shared_data = {}

        for F in (Home, MoveCamera, PreparingForMeasurement, Measurement, Measuring, GenitalMeasurement, StomachMeasurement): 
            page_name = F.__name__
            frame = F(parent=self.contanier, controller=self)
            self.frames[page_name] = frame
            #frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame("Home")

    def show_frame(self,page_name):
        frame = self.frames[page_name]
        frame.tkraise()

        if hasattr(frame,'on_page_open'):
            frame.on_page_open()

    def show_home(self):
        self.show_frame("Home")
        
    def show_move_camera(self):
        self.show_frame("MoveCamera")

    def show_preparing_for_measurement(self):
        self.show_frame("PreparingForMeasurement")

    def show_measurement(self):
        self.show_frame("Measurement")

    def show_measuring(self):
        self.show_frame("Measuring")

    def show_genital_measurement(self):
        self.show_frame("GenitalMeasurement")

    def show_stomach_measurement(self):
        self.show_frame("StomachMeasurement")

    def show_custom_popup(self,title,message):
        popup = Custom_popup(self, title, message)
        self.wait_window(popup.popup)
        
