from bluepy import btle
import struct

BEETLEMAC1 = '80:30:DC:D9:23:1E'

class Delegate(btle.DefaultDelegate):
    def __init__(self, MAC):
        btle.DefaultDelegate.__init__(self)
    
    def handleNotification(self, cHandle, data):
        global receivedData
        if(bytes(data) == b'A'): #handshake packet
            handShakeFlag[BEETLEMAC1]= True
        
        elif ('}' in data.decode('ISO-8859-1')): #detect the end of a packet
            receivedData += data
            charac= beetle.getCharacteristics(uuid = 'dfb1')[0]
            charac.write(bytes('A', 'ISO-8859-1'), withResponse=False)
            unpackedData = struct.unpack('<cc6Hc', receivedData)
            print(unpackedData)
            receivedData = bytes() #reset
        
        else:
            receivedData += data

def handShake(beetle):
    charac= beetle.getCharacteristics(uuid = 'dfb1')[0]
    charac.write(bytes('H', 'ISO-8859-1'), withResponse=False) #send handshake packet
    beetle.waitForNotifications(2)

    if(handShakeFlag[BEETLEMAC1] == True): 
        charac.write(bytes('A', 'ISO-8859-1'), withResponse=False)

def getData(beetle):
    while(1):
        beetle.waitForNotifications(2)       
    
def initSetup():
    beetle = btle.Peripheral(BEETLEMAC1)
    beetle_delegate = Delegate(BEETLEMAC1)
    beetle.withDelegate(beetle_delegate)
    return beetle

if __name__ == '__main__':
    handShakeFlag = {BEETLEMAC1: False}
    receivedData = bytes()
    beetle = initSetup()
    handShake(beetle)
    getData(beetle)
    
    

    

    
    