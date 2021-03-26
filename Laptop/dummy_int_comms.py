import time
import sys
from laptop_main import LaptopMain

#sending bytes to external comms timestamp and IMU data
def sendData():
    #rest at the start
    time.sleep(5)

    while(True):
        for i in range (5):
            #send dance data
            data = b'\x8a\x10\x00\x00N\xfe\xb6\xff\xe4\x1a\xff\xff\xf8\xff\x07\x00'
            #print(data)
            laptopMain.insert(data)
            time.sleep(0.33) #send at frequency around 30Hz
        
        #rest after a dance move
        time.sleep(5)

        #send position data
        laptopMain.position('R') 
        #print('R')
        
        #rest after a position change
        time.sleep(5)
                        
if __name__ == '__main__':

    laptopMain = LaptopMain()
    laptopMain.run()

    sendData()

    

    

    
    