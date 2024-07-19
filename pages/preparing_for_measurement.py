import tkinter as tk
from lib.camera import Camera

class PreparingForMeasurement(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.controller = controller

        tk.Label(self, text="測定の準備中です。").pack(pady=20)
        
        self.measurement_date_label = tk.Label(self, text="")
        self.measurement_date_label.pack(pady=5)
        
        self.capture_date_label = tk.Label(self, text="")
        self.capture_date_label.pack(pady=5)
        
        self.capture_location_label = tk.Label(self, text="")
        self.capture_location_label.pack(pady=5)
        
        self.species_label = tk.Label(self, text="")
        self.species_label.pack(pady=5)
        
    def on_page_open(self):
        self.measurement_date_label.config(text="測定日: "+self.controller.shared_data.get("measurement_date",""))
        self.capture_date_label.config(text="採集日: "+self.controller.shared_data.get("capture_date",""))
        self.capture_location_label.config(text="採集場所: "+self.controller.shared_data.get("capture_location",""))
        self.species_label.config(text="種: "+self.controller.shared_data.get("species",""))
        self.controller.update()
        
        Camera().move_to_distance(20)
        
        self.controller.show_measuring()
