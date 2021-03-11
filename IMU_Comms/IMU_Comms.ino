// I2C device class (I2Cdev) demonstration Arduino sketch for MPU6050 class using DMP (MotionApps v2.0)
// 6/21/2012 by Jeff Rowberg <jeff@rowberg.net>
// Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
//
// Changelog:
//      2019-07-08 - Added Auto Calibration and offset generator
//		   - and altered FIFO retrieval sequence to avoid using blocking code
//      2016-04-18 - Eliminated a potential infinite loop
//      2013-05-08 - added seamless Fastwire support
//                 - added note about gyro calibration
//      2012-06-21 - added note about Arduino 1.0.1 + Leonardo compatibility error
//      2012-06-20 - improved FIFO overflow handling and simplified read process
//      2012-06-19 - completely rearranged DMP initialization code and simplification
//      2012-06-13 - pull gyro and accel data from FIFO packet instead of reading directly
//      2012-06-09 - fix broken FIFO read sequence and change interrupt detection to RISING
//      2012-06-05 - add gravity-compensated initial reference frame acceleration output
//                 - add 3D math helper file to DMP6 example sketch
//                 - add Euler output and Yaw/Pitch/Roll output formats
//      2012-06-04 - remove accel offset clearing for better results (thanks Sungon Lee)
//      2012-06-01 - fixed gyro sensitivity to be 2000 deg/sec instead of 250
//      2012-05-30 - basic DMP initialization working

/* ============================================
I2Cdev device library code is placed under the MIT license
Copyright (c) 2012 Jeff Rowberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
*/

#include "I2Cdev.h"

#include "MPU6050_6Axis_MotionApps20.h"

#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

#include <FastCRC.h>

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for SparkFun breakout and InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 mpu;
//MPU6050 mpu(0x69); // <-- use for AD0 high

/* =========================================================================
   NOTE: In addition to connection 3.3v, GND, SDA, and SCL, this sketch
   depends on the MPU-6050's INT pin being connected to the Arduino's
   external interrupt #0 pin. On the Arduino Uno and Mega 2560, this is
   digital I/O pin 2.
 * ========================================================================= */

/* =========================================================================
   NOTE: Arduino v1.0.1 with the Leonardo board generates a compile error
   when using Serial.write(buf, len). The Teapot output uses this method.
   The solution requires a modification to the Arduino USBAPI.h file, which
   is fortunately simple, but annoying. This will be fixed in the next IDE
   release. For more info, see these links:

   http://arduino.cc/forum/index.php/topic,109987.0.html
   http://code.google.com/p/arduino/issues/detail?id=958
 * ========================================================================= */

#define INTERRUPT_PIN 2  // use pin 2 on Arduino Uno & most boards
#define LED_PIN 13 // (Arduino is 13, Teensy is 11, Teensy++ is 6)
bool blinkState = false;

// MPU control/status vars
bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer

// orientation/motion vars
Quaternion q;           // [w, x, y, z]         quaternion container
VectorInt16 aa;         // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
VectorInt16 aaWorld;    // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity;    // [x, y, z]            gravity vector
VectorInt16 gg;
VectorInt16 aaGravity;

float euler[3];         // [psi, theta, phi]    Euler angle container
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector

// packet structure for InvenSense teapot demo
uint8_t teapotPacket[14] = { '$', 0x02, 0,0, 0,0, 0,0, 0,0, 0x00, 0x00, '\r', '\n' };


// ================================================================
// ===               INTERRUPT DETECTION ROUTINE                ===
// ================================================================

volatile bool mpuInterrupt = false;     // indicates whether MPU interrupt pin has gone high
void dmpDataReady() {
    mpuInterrupt = true;
}


// ================================================================
// ===                      SETUP FOR COMMS                     ===
// ================================================================
FastCRC16 CRC16;

struct dataPacket {
  uint32_t beetleTime;
  bool startFlag;
  int16_t ac1;
  int16_t ac2;
  int16_t ac3;
  int16_t gy1;
  int16_t gy2;
  int16_t gy3;
};

struct dataPacket IMUPacket;

bool handshakeFlag = false;
bool firstDataRequest = false;
uint32_t baseTime = 0;
uint32_t preTime = 0;
uint32_t nowTime = 0;

// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================
bool isMoving = false;
bool changePosition = false;
bool isDancing = false;

const int sampleFrequency = 80; //40Hz sampling frequency rate
int counter = 0;
int16_t aaXPrevious = 0;
int16_t aaYPrevious = 0;
int16_t aaZPrevious = 0;
int16_t aaXDiff = 0;
int16_t aaYDiff = 0;
int16_t aaZDiff = 0;
int16_t aaXTotal = 0;
int16_t aaYTotal = 0;
int16_t aaZTotal = 0;

void setup() {
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
        Wire.setClock(400000); 
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    Serial.begin(115200);
    while (!Serial); 
    
    // initialize device
    mpu.initialize();
    pinMode(INTERRUPT_PIN, INPUT);

    devStatus = mpu.dmpInitialize();

    int bluno = 2;
    // set offsets
    if(bluno == 1)
    {
      mpu.setXGyroOffset(156);
      mpu.setYGyroOffset(-34);
      mpu.setZGyroOffset(56);
      mpu.setZAccelOffset(989); // 1688 factory default for my test chip
    }

    else if(bluno == 2)
    {
      mpu.setXGyroOffset(259);
      mpu.setYGyroOffset(-97);
      mpu.setZGyroOffset(19);
      mpu.setZAccelOffset(3578);
    }
     else if(bluno == 3)
    {
      mpu.setXGyroOffset(240);
      mpu.setYGyroOffset(-70);
      mpu.setZGyroOffset(15);
      mpu.setZAccelOffset(2472);
    }
    else if(bluno == 4)
    {
      mpu.setXGyroOffset(75);
      mpu.setYGyroOffset(-41);
      mpu.setZGyroOffset(-27);
      mpu.setZAccelOffset(3840);
    }
    else if(bluno == 5)
    {
      mpu.setXGyroOffset(71);
      mpu.setYGyroOffset(53);
      mpu.setZGyroOffset(-7);
      mpu.setZAccelOffset(3308);
    }
    else if(bluno == 6)
    {
      mpu.setXGyroOffset(169);
      mpu.setYGyroOffset(-67);
      mpu.setZGyroOffset(30);
      mpu.setZAccelOffset(2076);
    }
    else if(bluno == 7)
    {
      mpu.setXGyroOffset(259);
      mpu.setYGyroOffset(-97);
      mpu.setZGyroOffset(19);
      mpu.setZAccelOffset(3578);
    }

    if (devStatus == 0) 
    {
      // Calibration Time: generate offsets and calibrate our MPU6050
      mpu.CalibrateAccel(6);
      mpu.CalibrateGyro(6);
      mpu.PrintActiveOffsets();
      mpu.setDMPEnabled(true);

      // enable Arduino interrupt detection
      digitalPinToInterrupt(INTERRUPT_PIN);
      attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
      mpuIntStatus = mpu.getIntStatus();

      dmpReady = true;

      // get expected DMP packet size for later comparison
      packetSize = mpu.dmpGetFIFOPacketSize();
    } 
    else 
    {
        // ERROR!
    }

    // configure LED for output
    pinMode(LED_PIN, OUTPUT);
}



// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================

void loop() {
  
  if (!dmpReady) return;
    
  if(Serial.available())
  {
    char packet_type = Serial.read();

    if(packet_type == 'H')
    {
      Serial.write('A');
    }
    else if(packet_type == 'A')
    {
      handshakeFlag = true;
    }
    else if(packet_type == 'D' && handshakeFlag == true)
    {
      if(firstDataRequest == false)
      {
        firstDataRequest = true;
        baseTime = millis();
      }

      uint32_t timePassed;
      while(1)
      {
        preTime = millis();
        timePassed = millis();
        if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
             
          mpu.dmpGetAccel(&aaGravity, fifoBuffer);
          IMUPacket.ac1 = aaGravity.x;
          IMUPacket.ac2 = aaGravity.y;
          IMUPacket.ac3 = aaGravity.z;
          
          mpu.dmpGetGyro(&gg, fifoBuffer);
          IMUPacket.gy1 = gg.x;
          IMUPacket.gy2 = gg.y;
          IMUPacket.gy3 = gg.z;
        
          IMUPacket.beetleTime = timePassed - baseTime;
          Serial.write((const char *) &IMUPacket, sizeof(IMUPacket));
          uint16_t check = CRC16.modbus((uint8_t*)&IMUPacket, sizeof(IMUPacket));
          Serial.write((const char *) &check, sizeof(check));
          Serial.write('}');
        
          //delay(25);
          nowTime = millis();
          while(nowTime - preTime < 25)
          {
            nowTime = millis();
            if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
              
              mpu.dmpGetAccel(&aaGravity, fifoBuffer);
        
              aaXDiff = abs(aaGravity.x - aaXPrevious);
              aaXPrevious = aaGravity.x;
              
              aaYDiff = abs(aaGravity.y - aaYPrevious);
              aaYPrevious = aaGravity.y;
              
              aaZDiff = abs(aaGravity.z - aaZPrevious);
              aaZPrevious = aaGravity.z;
              

              if(aaXDiff > 400 && aaYDiff > 400 && aaZDiff > 400) {          
                isMoving = true;
                IMUPacket.startFlag = isMoving;
              } 

              
            }
          }
        }
      }
    }
  
    // blink LED to indicate activity
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
  }
}
