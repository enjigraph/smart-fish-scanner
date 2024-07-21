import csv
import pandas as pd
import os

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
