#include <FastCRC.h>

FastCRC16 CRC16;

struct dataPacket {
  uint32_t beetleTime;
  uint16_t ac1;
  uint16_t ac2;
  uint16_t ac3;
  uint16_t gy1;
  uint16_t gy2;
  uint16_t gy3;
};

struct dataPacket IMUPacket;

bool handshakeFlag = false;
bool firstDataRequest = false;
uint32_t baseTime = 0;

void setup() {
  Serial.begin(115200);
  
  IMUPacket.beetleTime = 0;
  IMUPacket.ac1 = 0;
  IMUPacket.ac2 = 200;
  IMUPacket.ac3 = 300;
  IMUPacket.gy1 = 400;
  IMUPacket.gy2 = 500;
  IMUPacket.gy3 = 60000;
}

void loop() {
  if(Serial.available())
  { 
    char packet_type = Serial.read();

    //handshake
    if(packet_type == 'H')
    {
      Serial.write('A');
    }
    else if (packet_type == 'A')
    {
      handshakeFlag = true;
    }
    //start transmission of data
    else if (packet_type == 'D' && handshakeFlag == true)
    {
      if(firstDataRequest == false)
      {
        firstDataRequest == true;
        baseTime = millis();
      }
      
      uint32_t timePassed;
      while(1)
      {
        timePassed = millis();
        IMUPacket.beetleTime = timePassed - baseTime;
        IMUPacket.ac1 += 1;
        Serial.write((const char *) &IMUPacket, sizeof(IMUPacket));
        uint16_t check = CRC16.modbus((uint8_t*)&IMUPacket, sizeof(IMUPacket));
        Serial.write((const char *) &check, sizeof(check));
        Serial.write('}');
        delay(25);
      }
    }
  }
}
