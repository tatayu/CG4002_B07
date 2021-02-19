from bluepy import btle
from bluepy.btle import BTLEException
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
                print(receivedPacket)
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
    charac.write(bytes('H'), withResponse=False) #send handshake packet
    beetle.waitForNotifications(2)

    if(handShakeFlag[BEETLEMAC1] == True): 
        charac.write(bytes('A'), withResponse=False)
        
def reconnect(beetle):
    success = False
    tryConnect = 0
    while(tryConnect < 2 and success == False):
        try:
            print('Reconnecting...')
            beetle.disconnect()
            beetle.connect(BEETLEMAC1)
            success = True

        except BTLEException:
            print('Failed to reconnect!')
            tryConnect += 1
    
    if(success == True):
        handShake(beetle)
    
    return success

def getData(beetle):
    global receivedPacket
    #start = time.time()
    while(1):
        try:
            beetle.waitForNotifications(2)
        
        except BTLEException:
            print('Device disconneted!')
            reconnect(beetle)

        '''
        end = time.time()
        if(end-start > 10):
            print(end-start)
            print(receivedPacket)
            break
        '''

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
    
    

    

    
    