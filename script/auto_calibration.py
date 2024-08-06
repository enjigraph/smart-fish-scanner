import serial
import time
import itertools

ser = serial.Serial('/dev/ttyACM2',9600)
time.sleep(2)

direction = ['stop','forward','backward','backward','forward']
angle = ['90,90','105,90','75,90','90,75','90,105','100,100','80,80']
patterns = list(itertools.product(angle,direction))
        
for pattern in patterns:

    instruction = ','.join(map(str,pattern))
    print(f'{instruction}')
    ser.write(instruction.encode())
    time.sleep(4)
            
