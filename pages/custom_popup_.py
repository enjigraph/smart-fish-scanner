import tkinter as tk

class CustomPopup():
    def __init__(self, parent, title, message):
        self.parent = parent
        self.title = title
        self.message = message

        popup = tk.Toplevel()
        popup.title(self.title)

        label = tk.Label(popup, text=self.message,padx=20, pady=20)
        label.pack()

        ok_button = tk.Button(popup,text="OK", command=popup.destroy)
        ok_button.pack(pady=10)
        
        popup.geometry("300x200")

        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
        y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
        popup.geometry(f"+{x}+{y}")

        popup.transient(self.parent)
        popup.grab_set()

    def close_popup():
        self.popup.destroy()
    
        
