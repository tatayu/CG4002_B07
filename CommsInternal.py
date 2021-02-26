from bluepy import btle
from bluepy.btle import BTLEException
from datetime import datetime
import struct
import time
import threading
from crccheck.crc import Crc16, CrcModbus

BEETLEMAC1 = '80:30:DC:E9:1C:2F'
BEETLEMAC2 = '80:30:DC:D9:1C:60'

class Delegate(btle.DefaultDelegate):
    def __init__(self, BEETLEMAC):
        btle.DefaultDelegate.__init__(self)
        self.BEETLEMAC = BEETLEMAC
    
    def handleNotification(self, cHandle, data):
        global receivedData
        global missPakcet
        global receivedPacket
        if(data == b'A'): #handshake packet
            handShakeFlag[self.BEETLEMAC]= True

        elif (b'}' in data): #detect the end of a packet
            try:   
                receivedData += data[0:data.index(b'}')+1]
                unpackedData = struct.unpack('<6H', receivedData[0:len(receivedData)-3])
                receivedPacket += 1
                beetleCrc = struct.unpack('<H', receivedData[(len(receivedData)-3):(len(receivedData)-1)])
                pcCrc = CRC(beetleCrc, receivedData)
                #print(beetleCrc)

                #if(pcCrc == beetleCrc):
                dataBuffer[self.BEETLEMAC] = unpackedData
                print(self.BEETLEMAC, dataBuffer[self.BEETLEMAC])
                receivedData = bytes() #reset
                receivedData += data[data.index(b'}')+1:len(data)]
        
            except Exception as e:
                missPakcet += 1
                print(missPakcet)
                receivedData = bytes()
                receivedData += data[data.index(b'}')+1:len(data)]

        else:
            receivedData += data

class beetleThread (threading.Thread):
    def __init__(self, threadID, BEETLEMAC):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.BEETLEMAC = BEETLEMAC
    
    def run(self):
        beetle = initSetup(self.BEETLEMAC)
        handShake(beetle, self.BEETLEMAC)
        getData(beetle, self.BEETLEMAC)

def CRC(beetleCrc, receivedData):
    crcCheck = CrcModbus()
    crcCheck.process(receivedData[0:len(receivedData)-3])
    
    return crcCheck.final()

def handShake(beetle, BEETLEMAC):
    print('Handshaking...')
    charac= beetle.getCharacteristics(uuid = 'dfb1')[0]
    charac.write(bytes('H', 'ISO 8859-1'), withResponse=False) #send handshake packet
    beetle.waitForNotifications(2)

    if(handShakeFlag[BEETLEMAC] == True): 
        charac.write(bytes('A', 'ISO 8859-1'), withResponse=False)

'''       
def sendTimestamp(beetle):
    charac= beetle.getCharacteristics(uuid = 'dfb1')[0]
    charac.write(bytes('T', 'ISO 8859-1'), withResponse=False)
    beetle.waitForNotifications(2)

    if(timestampFlag[BEETLEMAC1] == True):
        timestamp = datetime.now().time()
        print(timestamp)
        #str(timestamp)
        charac.write(bytes(str(timestamp), 'ISO 8859-1'), withResponse = False)
        #beetle.waitForNotifications(2)
'''

def reconnect(beetle, BEETLEMAC):
    success = False
    tryConnect = 0
    while(tryConnect < 2 and success == False):
        try:
            print('Reconnecting...')
            beetle.disconnect()
            beetle.connect(BEETLEMAC)
            success = True

        except BTLEException:
            print('Failed to reconnect!')
            tryConnect += 1
    
    if(success == True):
        handShake(beetle, BEETLEMAC)
    
    return success

def getData(beetle, BEETLEMAC):
    global receivedPacket
    #start = time.time()
    while(1):
        try:
            beetle.waitForNotifications(2)
        
        except BTLEException:
            print('Device disconneted!')
            reconnect(beetle, BEETLEMAC)

        '''
        end = time.time()
        if(end-start > 10):
            print(end-start)
            print(receivedPacket)
            break
        '''

def initSetup(BEETLEMAC):
    beetle = btle.Peripheral(BEETLEMAC)
    beetle_delegate = Delegate(BEETLEMAC)
    beetle.withDelegate(beetle_delegate)
    return beetle

if __name__ == '__main__':
    #print(datetime.now().time())
    handShakeFlag = {BEETLEMAC1: False, BEETLEMAC2: False}
    dataBuffer = {BEETLEMAC1: "", BEETLEMAC2: ""}
    missPakcet = 0
    receivedPacket = 0
    receivedData = bytes()
    #thread1 = beetleThread(1, BEETLEMAC1)
    thread2 = beetleThread(2, BEETLEMAC2)
    #thread1.start()
    thread2.start()
    
    

    

    
    