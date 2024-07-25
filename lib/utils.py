import csv
import pandas as pd
import os
import cv2
from pyzbar.pyzbar import decode

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f'{folder_path}が作成されました。')
    else:
        print(f'{folder_path}は既に存在しています。')      

def count_column_elements(file_path,column_name):
    try:
        df = pd.read_csv(file_path)
        return df[column_name].count()
    except:
        return 0

def get_qr_code_data(frame):
    detector = cv2.QRCodeDetectorAruco()

    height, width = frame.shape[:2]
    
    data, _, _ = detector.detectAndDecode(frame)
    print(f'qr code data: {data}')

    cv2.imshow('qr_code',frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    if data:
        return data
            
    return None
