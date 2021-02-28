from bluepy import btle
from bluepy.btle import BTLEException
from datetime import datetime
import struct
import time
import threading
import sys
from crccheck.crc import Crc16, CrcModbus

BEETLEMAC1 = '80:30:DC:E9:1C:2F'
BEETLEMAC2 = '80:30:DC:E8:EF:FA'
BEETLEMAC3 = '34:B1:F7:D2:34:71'

class Delegate(btle.DefaultDelegate):
    def __init__(self, BEETLEMAC):
        btle.DefaultDelegate.__init__(self)
        self.BEETLEMAC = BEETLEMAC
    
    def handleNotification(self, cHandle, data):
        #receive handshake reply from beetle
        if(data == b'A'): 
            handShakeFlag[self.BEETLEMAC]= True
        
        #detect the end of a packet
        elif (b'}' in data): 
            try:   
                receivedData[self.BEETLEMAC] += data[0:data.index(b'}')+1]
                unpackedData, beetleCrc = unpackPacket(receivedData, self.BEETLEMAC) 
                pcCrc = CRC(beetleCrc, receivedData, self.BEETLEMAC)

                #Compare CRC calculated by PC and beetle
                if(str(pcCrc) == str(beetleCrc)[1:len(str(beetleCrc))-2]):
                    timestamp = timeParse(self.BEETLEMAC, unpackedData)

                    print(self.BEETLEMAC,timestamp)
                    print("real time: ",getTime() )
                    dataBuffer[self.BEETLEMAC] = unpackedData[1:]
                    print(self.BEETLEMAC, dataBuffer[self.BEETLEMAC])
                    receivedPacket[self.BEETLEMAC] += 1
                else:
                    print('CRC Failed!')
                    missedPacket[self.BEETLEMAC] += 1

                #reset
                receivedData[self.BEETLEMAC] = bytes() 
                receivedData[self.BEETLEMAC] += data[data.index(b'}')+1:len(data)]
        
            except Exception as e:
                missedPacket[self.BEETLEMAC] += 1
                print(missedPacket[self.BEETLEMAC])
                receivedData[self.BEETLEMAC] = bytes()
                receivedData[self.BEETLEMAC] += data[data.index(b'}')+1:len(data)]

        else:
            receivedData[self.BEETLEMAC] += data

class beetleThread (threading.Thread):
    def __init__(self, threadID, BEETLEMAC):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.BEETLEMAC = BEETLEMAC
    
    def run(self):
        getIMUData(self.BEETLEMAC)

def unpackPacket(receivedData, BEETLEMAC):
    unpackedData = struct.unpack('<I6H', receivedData[BEETLEMAC][0:len(receivedData[BEETLEMAC])-3])            
    beetleCrc = struct.unpack('<H', receivedData[BEETLEMAC][(len(receivedData[BEETLEMAC])-3):(len(receivedData[BEETLEMAC])-1)])
    return unpackedData, beetleCrc

def getTime():
    currentTime = datetime.now().time()
    timeDecimal = currentTime.hour*3600*1000 + currentTime.minute*60*1000 + currentTime.second*1000 + round(currentTime.microsecond/1000)
    return timeDecimal

def timeParse(BEETLEMAC, unpackedData):
    milliTime = unpackedData[0]
    if(reconnectFlag[BEETLEMAC] == True):
        return milliTime + reconnectTimestamp[BEETLEMAC]
    else:
        return milliTime + startTimestamp[BEETLEMAC]

def CRC(beetleCrc, receivedData, BEETLEMAC):
    crcCheck = CrcModbus()
    crcCheck.process(receivedData[BEETLEMAC][0:len(receivedData[BEETLEMAC])-3])
    return crcCheck.final()

def handShake(BEETLEMAC):
    try:
        print('Handshaking with ', BEETLEMAC, '...')
        charac= beetleObject[BEETLEMAC].getCharacteristics(uuid = 'dfb1')[0]
        charac.write(bytes('H', 'ISO 8859-1'), withResponse=False) 
        beetleObject[BEETLEMAC].waitForNotifications(2)

        if(handShakeFlag[BEETLEMAC] == True): 
            charac.write(bytes('A', 'ISO 8859-1'), withResponse=False)
            print('Handshake done with ', BEETLEMAC, '!')
    
    except BTLEException:
        print('Handshake with ', BEETLEMAC,' failed!')
        reconnect(BEETLEMAC)

def reconnect(BEETLEMAC):
    success = False
    tryConnect = 0
    reconnectFlag[BEETLEMAC] = False
    while(tryConnect < 2 and success == False):
        try:
            print('Reconnecting...')
            beetleObject[BEETLEMAC].disconnect()
            beetleObject[BEETLEMAC].connect(BEETLEMAC)
            success = True

        except BTLEException:
            print('Failed to reconnect!')
            tryConnect += 1
    
    if(success == True):
        reconnectFlag[BEETLEMAC] = True
        handShake(BEETLEMAC)

    return success

def getIMUData(BEETLEMAC):
    reconnectTimeFlag = False
    charac= beetleObject[BEETLEMAC].getCharacteristics(uuid = 'dfb1')[0]
    charac.write(bytes('D', 'ISO 8859-1'), withResponse=False) 
 
    timestamp = getTime()
    startTimestamp[BEETLEMAC] = timestamp
    print('Ready to start: ',timestamp)

    while(1):
        try:
            if(reconnectTimeFlag == True):
                timestamp = getTime()
                reconnectTimestamp[BEETLEMAC] = timestamp
                print('Reconnect ready to start: ', timestamp)
                reconnectTimeFlag == False
            
            if not beetleObject[BEETLEMAC].waitForNotifications(2):
                charac= beetleObject[BEETLEMAC].getCharacteristics(uuid = 'dfb1')[0]
                charac.write(bytes('D', 'ISO 8859-1'), withResponse=False)
                IMUDataRequest[BEETLEMAC] += 1

                if(IMUDataRequest[BEETLEMAC] > 4):
                    reconnect(BEETLEMAC)
                    IMUDataRequest[BEETLEMAC] = 0 #reset
        
        except BTLEException:
            print('Device disconneted!')
            reconnect(BEETLEMAC)
            reconnectTimeFlag = True

        '''
        end = time.time()
        if(end-start > 10):
            print(end-start)
            print(receivedPacket)
            break
        '''

def initSetup(BEETLEMAC):
    initSetupSuccess[BEETLEMAC] = False
    try:
        if(initSetupSuccess[BEETLEMAC] == False and initSetupRetry[BEETLEMAC] < 3):
            beetle = btle.Peripheral(BEETLEMAC)
            beetle_delegate = Delegate(BEETLEMAC)
            beetle.withDelegate(beetle_delegate)
            initSetupSuccess[BEETLEMAC] = True
            beetleObject[BEETLEMAC] = beetle

        else:
            print('Failed to connect beetle ', BEETLEMAC)
            beetleIDList.remove(BEETLEMAC)
            return
     
    except BTLEException:
            print('Initial setup fialed! Please disconnect the bluetooth for beetle ', BEETLEMAC, ' and try again!')
            initSetupRetry[BEETLEMAC] += 1
            initSetup(BEETLEMAC)


if __name__ == '__main__':
    beetleIDList = [BEETLEMAC1, BEETLEMAC2, BEETLEMAC3]
    beetleObject = {}

    #If initial setup is success, initSetupSuccess is set to true
    initSetupSuccess = {BEETLEMAC1: False, BEETLEMAC2: False, BEETLEMAC3: False}

    #Number of retries if initial setup fails
    initSetupRetry = {BEETLEMAC1: 0, BEETLEMAC2: 0, BEETLEMAC3: 0}
    
    #If receive reply from beetle, handShakeFlag is set to true
    handShakeFlag = {BEETLEMAC1: False, BEETLEMAC2: False, BEETLEMAC3: False}
    
    #Number of times of request for data from beetle
    IMUDataRequest = {BEETLEMAC1: 0, BEETLEMAC2: 0, BEETLEMAC3: 0}

    #received data in bytes
    receivedData = {BEETLEMAC1: bytes(), BEETLEMAC2: bytes(), BEETLEMAC3: bytes()}

    #number of missed packets
    missedPacket = {BEETLEMAC1: 0, BEETLEMAC2: 0, BEETLEMAC3: 0}

    #number of received packets
    receivedPacket = {BEETLEMAC1: 0, BEETLEMAC2: 0, BEETLEMAC3: 0}

    #Store the transferred from 3 beetles
    dataBuffer = {BEETLEMAC1: "", BEETLEMAC2: "", BEETLEMAC3: ""}

    #Store the real time in decimal millisecond at the start of handshake
    startTimestamp = {BEETLEMAC1: 0, BEETLEMAC2: 0, BEETLEMAC3: 0}

    #If reconnect happens and it's success, reconnectFlag is set to true
    reconnectFlag = {BEETLEMAC1: False, BEETLEMAC2: False, BEETLEMAC3: False}

    #Store the real time in decimal millisecond when the reconnection is successful
    reconnectTimestamp = {BEETLEMAC1: 0, BEETLEMAC2: 0, BEETLEMAC3: 0}
    
    for BEETLEMAC in beetleIDList:
        initSetup(BEETLEMAC)
        if(initSetupSuccess[BEETLEMAC] == True):
            handShake(BEETLEMAC)

    thread1 = beetleThread(1, BEETLEMAC1)
    thread2 = beetleThread(2, BEETLEMAC2)
    thread3 = beetleThread(3, BEETLEMAC3)

    thread1.start()
    thread2.start()
    thread3.start()
    
    

    

    
    