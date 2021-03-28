from bluepy import btle
from bluepy.btle import BTLEException
from datetime import datetime
import struct
import time
import threading
import sys
import json
from crccheck.crc import Crc16, CrcModbus
from laptop_main import LaptopMain

BEETLEMAC1 = '80:30:DC:E9:1C:2F'
#BEETLEMAC1 = '34:B1:F7:D2:34:71'
#BEETLEMAC2 = '80:30:DC:D9:23:1E'
BEETLEMAC2 = '80:30:DC:E8:EF:FA'

counter = 0
dataCollection = []

def handShake(BEETLEMAC):
    try_num = 0
    try:
        while(try_num < 5):
            print('Handshaking with ', beetleName[BEETLEMAC], '...')
            charac= beetleObject[BEETLEMAC].getCharacteristics(uuid = 'dfb1')[0]
            charac.write(bytes('H', 'ISO 8859-1'), withResponse=False) 
            beetleSendTime[BEETLEMAC] = getTime()
            print('sending H to ', beetleName[BEETLEMAC])
            beetleObject[BEETLEMAC].waitForNotifications(2)

            if(handShakeFlag[BEETLEMAC] == True): 
                charac.write(bytes('A', 'ISO 8859-1'), withResponse=False)
                print('sending A to ', beetleName[BEETLEMAC])
                handShakeSuccess[BEETLEMAC] = True
                print('Handshake done with ', beetleName[BEETLEMAC], '!')
                break
            
            try_num += 1

    except BTLEException:
        print('Handshake with ', beetleName[BEETLEMAC],' failed!')
        reconnect(BEETLEMAC)

def reconnect(BEETLEMAC):
    success = False
    tryConnect = 0
    reconnectFlag[BEETLEMAC] = False
    while(tryConnect < 2 and success == False):
        try:
            print('Reconnecting...', beetleName[BEETLEMAC])
            beetleObject[BEETLEMAC].disconnect()
            beetleObject[BEETLEMAC].connect(BEETLEMAC)
            success = True

        except BTLEException:
            print('Failed to reconnect!', beetleName[BEETLEMAC])
            tryConnect += 1
    
    if(success == True):
        reconnectFlag[BEETLEMAC] = True
        handShake(BEETLEMAC)

    return success

def getIMUData(BEETLEMAC):
    reconnectTimeFlag = False
    charac= beetleObject[BEETLEMAC].getCharacteristics(uuid = 'dfb1')[0]
    charac.write(bytes('D', 'ISO 8859-1'), withResponse=False) 
    print('sending D to beetle ', beetleName[BEETLEMAC])
    
    timestamp = getTime()
    startTimestamp[BEETLEMAC] = timestamp
    print('Ready to start: ',beetleName[BEETLEMAC], timestamp)

    while(1):
        try:
            if not beetleObject[BEETLEMAC].waitForNotifications(2):
                charac= beetleObject[BEETLEMAC].getCharacteristics(uuid = 'dfb1')[0]
                print('sending D to beetle ', beetleName[BEETLEMAC])
                charac.write(bytes('D', 'ISO 8859-1'), withResponse=False)

                if(reconnectTimeFlag == True):
                    timestamp = getTime()
                    reconnectTimestamp[BEETLEMAC] = timestamp
                    print('Reconnect ready to start: ', beetleName[BEETLEMAC], timestamp)
                    reconnectTimeFlag = False
            
                IMUDataRequest[BEETLEMAC] += 1

                if(IMUDataRequest[BEETLEMAC] > 4):
                    reconnect(BEETLEMAC)
                    IMUDataRequest[BEETLEMAC] = 0 #reset
                
        except BTLEException:
            print('Device disconneted!', beetleName[BEETLEMAC]) 
            reconnectTimeFlag = reconnect(BEETLEMAC)

def unpackPacket(receivedData, BEETLEMAC):
    unpackedData = struct.unpack('<?I6h', receivedData[BEETLEMAC][0:len(receivedData[BEETLEMAC])-3])            
    beetleCrc = struct.unpack('<H', receivedData[BEETLEMAC][(len(receivedData[BEETLEMAC])-3):(len(receivedData[BEETLEMAC])-1)])
    return unpackedData, beetleCrc

def getTime():
    currentTime = datetime.now().time()
    timeDecimal = currentTime.hour*3600*1000 + currentTime.minute*60*1000 + currentTime.second*1000 + round(currentTime.microsecond/1000)
    return timeDecimal

def timeParse(BEETLEMAC, unpackedData):
    milliTime = unpackedData[1]
    if(reconnectFlag[BEETLEMAC] == True):
        return milliTime + reconnectTimestamp[BEETLEMAC] + beetleOffset[BEETLEMAC]
    else:
        return milliTime + startTimestamp[BEETLEMAC] + beetleOffset[BEETLEMAC]

def CRC(beetleCrc, receivedData, BEETLEMAC):
    crcCheck = CrcModbus()
    crcCheck.process(receivedData[BEETLEMAC][0:len(receivedData[BEETLEMAC])-3])
    return crcCheck.final()

class Delegate(btle.DefaultDelegate):
    def __init__(self, BEETLEMAC):
        btle.DefaultDelegate.__init__(self)
        self.BEETLEMAC = BEETLEMAC
    
    def handleNotification(self, cHandle, data):
        #receive handshake reply from beetle
        if(data == b'A'): 
            beetleReceiveTime[self.BEETLEMAC] = getTime()
            print('received A from ', beetleName[self.BEETLEMAC])
            handShakeFlag[self.BEETLEMAC]= True
            beetleOffset[self.BEETLEMAC] = round((beetleReceiveTime[self.BEETLEMAC] - beetleSendTime[self.BEETLEMAC])/2)
            print(beetleName[self.BEETLEMAC], 'Offset: ', beetleOffset[self.BEETLEMAC])
        
        #detect the end of a packet
        elif (b'}' in data): 
            
            try:   
                receivedData[self.BEETLEMAC] += data[0:data.index(b'}')+1]
                
                #get waist data
                if(beetleName[self.BEETLEMAC] == 'beetle2'):
                    unpackedData = struct.unpack('<cc', receivedData[self.BEETLEMAC])
                    direction = unpackedData[0].decode('ISO-8859-1')
                    if(direction == 'N'):
                        print('Stationary...')
                    else:
                        #sending position to ext comms
                        laptopMain.position(direction)
                        print(beetleName[self.BEETLEMAC], 'data: ', direction)
                
                #get arm data
                else:
                    unpackedData, beetleCrc = unpackPacket(receivedData, self.BEETLEMAC) 
                    pcCrc = CRC(beetleCrc, receivedData, self.BEETLEMAC)

                    #Compare CRC calculated by PC and beetle
                    if(str(pcCrc) == str(beetleCrc)[1:len(str(beetleCrc))-2]):
                        timestamp = timeParse(self.BEETLEMAC, unpackedData)
                        #print(getTime() - beetleOffset[self.BEETLEMAC])
                        print(beetleName[self.BEETLEMAC], 'timestamp: ' ,timestamp)
                        dataBuffer[self.BEETLEMAC] = unpackedData[1:]
                        
                        #dnace movement is false
                        if(unpackedData[0] == False):
                            print('Resting...')
                        
                        #dance movement is true
                        else: 
                            #sending bytes to external comms timestamp and IMU data
                            sendData = struct.pack('<I6h', timestamp, unpackedData[2], unpackedData[3], unpackedData[4], unpackedData[5], unpackedData[6], unpackedData[7])
                            laptopMain.insert(sendData)
                            #print(beetleName[self.BEETLEMAC], 'data: ', sendData)
                            print(beetleName[self.BEETLEMAC], 'data: ', str(unpackedData[1:]))

                        receivedPacket[self.BEETLEMAC] += 1

                    else:
                        print(beetleName[self.BEETLEMAC], 'CRC Failed!')
                        missedPacket[self.BEETLEMAC] += 1
                        print(beetleName[self.BEETLEMAC], 'misspacket: ', missedPacket[self.BEETLEMAC])
                        
                #reset
                receivedData[self.BEETLEMAC] = bytes() 
                receivedData[self.BEETLEMAC] += data[data.index(b'}')+1:len(data)]
        
            except Exception as e:
                #print(e)
                missedPacket[self.BEETLEMAC] += 1
                print(beetleName[self.BEETLEMAC], 'misspacket: ', missedPacket[self.BEETLEMAC])
                
                #reset
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
        if(handShakeSuccess[self.BEETLEMAC]):
            getIMUData(self.BEETLEMAC)
        else:
            beetleIDList.remove(BEETLEMAC)
            
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
            print('Failed to connect beetle ', beetleName[BEETLEMAC])
            beetleIDList.remove(BEETLEMAC)
            return
    
    except BTLEException:
        print('Initial setup failed! Please disconnect the bluetooth for beetle ', beetleName[BEETLEMAC], ' and try again!')
        initSetupRetry[BEETLEMAC] += 1
        initSetup(BEETLEMAC)

if __name__ == '__main__':

    beetleIDList = [BEETLEMAC2]
    beetleObject = {}

    beetleName = {BEETLEMAC1: "beetle1", BEETLEMAC2: "beetle2"}

    beetleSendTime = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    beetleReceiveTime = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    beetleOffset = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    #If initial setup is success, initSetupSuccess is set to true
    initSetupSuccess = {BEETLEMAC1: False, BEETLEMAC2: False}

    #Number of retries if initial setup fails
    initSetupRetry = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    #If receive reply from beetle, handShakeFlag is set to true
    handShakeFlag = {BEETLEMAC1: False, BEETLEMAC2: False}

    #True if the handShake is success
    handShakeSuccess = {BEETLEMAC1: False, BEETLEMAC2:False}

    #Number of times of request for data from beetle
    IMUDataRequest = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    #received data in bytes
    receivedData = {BEETLEMAC1: bytes(), BEETLEMAC2: bytes()}

    #number of missed packets
    missedPacket = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    #number of received packets
    receivedPacket = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    #Store the transferred from 3 beetles
    dataBuffer = {BEETLEMAC1: "", BEETLEMAC2: ""}

    #Store the real time in decimal millisecond at the start of handshake
    startTimestamp = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    #If reconnect happens and it's success, reconnectFlag is set to true
    reconnectFlag = {BEETLEMAC1: False, BEETLEMAC2: False}

    #Store the real time in decimal millisecond when the reconnection is successful
    reconnectTimestamp = {BEETLEMAC1: 0, BEETLEMAC2: 0}

    laptopMain = LaptopMain()
    laptopMain.run()

    for BEETLEMAC in beetleIDList:
        initSetup(BEETLEMAC)
        if(initSetupSuccess[BEETLEMAC] == True):
            handShake(BEETLEMAC)

    #thread1 = beetleThread(1, BEETLEMAC1)
    thread2 = beetleThread(2, BEETLEMAC2)
    
    #thread1.start()
    thread2.start()

    

    

    
    