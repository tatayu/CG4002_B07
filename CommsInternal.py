from bluepy import btle
import struct
import time

BEETLEMAC1 = '80:30:DC:D9:23:1E'

class Delegate(btle.DefaultDelegate):
    def __init__(self, MAC):
        btle.DefaultDelegate.__init__(self)
    
    def handleNotification(self, cHandle, data):
        global receivedData
        global missPakcet
        global receivedPacket
        if(data == b'A'): #handshake packet
            handShakeFlag[BEETLEMAC1]= True

        elif (b'}' in data): #detect the end of a packet
            try:
                receivedData += data[0:data.index(b'}')+1]
                unpackedData = struct.unpack('<c6Hc', receivedData)
                receivedPacket += 1
                #print(unpackedData)
                receivedData = bytes() #reset
                receivedData += data[data.index(b'}')+1:len(data)]
        
            except Exception as e:
                missPakcet += 1
                print(missPakcet)
                receivedData = bytes()
                receivedData += data[data.index(b'}')+1:len(data)]

        else:
            receivedData += data

def handShake(beetle):
    charac= beetle.getCharacteristics(uuid = 'dfb1')[0]
    charac.write(bytes('H', 'ISO-8859-1'), withResponse=False) #send handshake packet
    beetle.waitForNotifications(2)

    if(handShakeFlag[BEETLEMAC1] == True): 
        charac.write(bytes('A', 'ISO-8859-1'), withResponse=False)

def getData(beetle):
    global receivedPacket
    start = time.time()
    while(1):
        beetle.waitForNotifications(2)
        end = time.time()
        if(end-start > 10):
            print(end-start)
            print(receivedPacket)
            break

def initSetup():
    beetle = btle.Peripheral(BEETLEMAC1)
    beetle_delegate = Delegate(BEETLEMAC1)
    beetle.withDelegate(beetle_delegate)
    return beetle

if __name__ == '__main__':
    handShakeFlag = {BEETLEMAC1: False}
    missPakcet = 0
    receivedPacket = 0
    receivedData = bytes()
    beetle = initSetup()
    handShake(beetle)
    getData(beetle)
    
    

    

    
    