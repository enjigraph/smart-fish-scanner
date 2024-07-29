import serial
import time
import sys
import threading
import re
import numpy as np

class DigitalScale:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DigitalScale, cls).__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, *args, **kwargs):
        self.ser = serial.Serial('/dev/ttyUSB0', 9600)
        time.sleep(2)
        
        self.data = None
        self.stable_data = []
        self.recent_data = []
        self.running = True
        self.thread = threading.Thread(target=self._update_data)
        self.thread.daemon = True
        self.thread.start()

    def _update_data(self):
        while self.running:
            if self.ser.in_waiting>0:
                try:
                    line = self.ser.readline().decode('utf-8').rstrip()
                    #print(line)
                    match = re.search(r"[0-9]+\.?[0-9]*",line)

                    if match:
                        #print(f'weight: {match.group()}')
                        self.data = match.group()
                        self.recent_data.append(float(match.group()))

                        if len(self.recent_data) > 5:
                            self.recent_data.pop(0)
                        
                    else:
                        print('weight is not match')
    
                except Exception as err:
                    print(f'[DigitalScale][error] {err}')
                    self.data = None
            else:
                self.data = None

            time.sleep(0.1)

    def get_data(self):
        #print("weight:"+str(self.data))

        if not self.data:
            return None
        
        return float(self.data) 

    def update_stable_data(self):

        if len(self.recent_data) < 5:
            return 0
        
        if np.std(self.recent_data[-5:]) < 0.1:

            mean_value = np.mean(self.recent_data)
            
            if not self.stable_data or abs(mean_value - self.stable_data[-1]) >= 5:
                self.stable_data.append(np.mean(self.recent_data))
                
            self.recent_data.clear()

        return 0
    
    def get_stable_data(self):
        return self.stable_data

    def clear_stable_data(self):
        self.stable_data.clear()
        return 0

    def get_stable_data_length(self):
        return len(self.stable_data)

    def get_weight(self):
        print(self.stable_data)
        max_stable_data = max(self.stable_data)
        self.stable_data.remove(max_stable_data)
        second_max_stable_data = max(self.stable_data)

        return max_stable_data - second_max_stable_data
    
    def stop(self):
        self.running = False
