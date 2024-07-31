import serial
import time

ser = serial.Serial('/dev/ttyACM2',9600,timeout=2)
time.sleep(2)

def move(angle1,angle2,direction):

    ser.write(f'{angle1},{angle2},{direction}\n'.encode())
    time.sleep(1)
    
    ser.close()

move(90,90,'forward')
