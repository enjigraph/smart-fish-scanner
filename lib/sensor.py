import serial
import time
import sys
import threading

class Sensor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Sensor, cls).__new__(cls)
                    cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self, *args, **kwargs):
        self.ser = serial.Serial('/dev/ttyACM1', 115200)
        time.sleep(2)
        self.data = (None,None)
        self.running = True
        self.thread = threading.Thread(target=self._update_data)
        self.thread.daemon = True
        self.thread.start()

    def _update_data(self):
        while self.running:

            self.ser.reset_input_buffer()
            time.sleep(0.5)

            if self.ser.in_waiting>0:
                try:
                    line = self.ser.readline().decode('utf-8').rstrip()
                    
                    ir_value, distance = line.split(',')
                
                    #print(f'IR Value: {ir_value}, Distance: {distance}')

                    self.data = (ir_value, distance)
                except:
                    pass
                    #print('sensor error')
                    #self.data = (None,None)
            
    def get_data(self):
        return self.data

    def stop(self):
        self.running = False
        self.thread.join()
