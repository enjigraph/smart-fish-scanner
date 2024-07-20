import tkinter as tk
from tkcalendar import Calendar, DateEntry

class Measurement(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.controller = controller
        
        tk.Label(self, text="メタデータを入力してください。まだイワシは置かないでください。").pack(pady=20)

        tk.Label(self, text="測定日").pack(pady=5)
        
        self.measurement_date_entry = DateEntry(self,width=12,background='darkblue', foreground='white',borderwidth=2)
        self.measurement_date_entry.pack(pady=5)
        
        tk.Label(self, text="採集日").pack(pady=5)
        self.capture_date_entry = DateEntry(self,width=12,background='darkblue', foreground='white',borderwidth=2)
        self.capture_date_entry.pack(pady=5)
        
        tk.Label(self, text="採集場所").pack(pady=5)
        self.capture_location_entry = tk.Entry(self)
        self.capture_location_entry.pack(pady=5)
        
        tk.Label(self, text="種").pack(pady=5)
        self.species_entry = tk.Entry(self)
        self.species_entry.pack(pady=5)
        
        tk.Button(self,text="測定を開始する",command=self.start).pack(pady=10)
        tk.Button(self,text="戻る",command=self.controller.show_home).pack(pady=30)

    def start(self):
        self.controller.shared_data["measurement_date"] = self.measurement_date_entry.get()
        self.controller.shared_data["capture_date"] = self.capture_date_entry.get()
        self.controller.shared_data["capture_location"] = self.capture_location_entry.get()
        self.controller.shared_data["species"] = self.species_entry.get()
        
        self.controller.show_preparing_for_measurement()
        
