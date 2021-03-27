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

#include<WiFi.h>

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for SparkFun breakout and InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 mpu;
//MPU6050 mpu(0x69); // <-- use for AD0 high

byte mac[6];

#define OUTPUT_ACCEL_GRAVITY //Acc Values with gravity

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
VectorInt16 aaGravity;
VectorInt16 aag;


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
bool handshakeFlag = false;

// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================

bool isMoving = true;
bool changePosition = false;
bool isDancing = false;

const int sampleFrequency = 100; //40Hz sampling frequency rate
int counter = 0;
int valueCounterX = 0;
int valueCounterY = 0;
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
    
    //CHANGE BEFORE UPLAOD TO BEETLE
    int bluno = 6;
    
    if(bluno == 1) //Arm Beetle 1
    {
      mpu.setXGyroOffset(216);
      mpu.setYGyroOffset(-40);
      mpu.setZGyroOffset(33);
      mpu.setZAccelOffset(597); // 1688 factory default for my test chip
    }

    else if(bluno == 2) //Arm Beetle 2
    {
      mpu.setXGyroOffset(253);
      mpu.setYGyroOffset(-103);
      mpu.setZGyroOffset(8);
      mpu.setZAccelOffset(1804);
    }
     else if(bluno == 3) //Arm Beetle 3
    {
      mpu.setXGyroOffset(81);
      mpu.setYGyroOffset(-25);
      mpu.setZGyroOffset(-34);
      mpu.setZAccelOffset(2122);
    }
    else if(bluno == 4) //Waist Beetle 1
    {
      mpu.setXGyroOffset(155);
      mpu.setYGyroOffset(-37);
      mpu.setZGyroOffset(-56);
      mpu.setZAccelOffset(1006);
    }
    else if(bluno == 5) // Waist Beetle 2
    {
      mpu.setXGyroOffset(254);
      mpu.setYGyroOffset(-55);
      mpu.setZGyroOffset(18);
      mpu.setZAccelOffset(973);
    }
    else if(bluno == 6) //Waist Beetle 3
    {
      mpu.setXGyroOffset(80);
      mpu.setYGyroOffset(36);
      mpu.setZGyroOffset(-19);
      mpu.setZAccelOffset(1770);
    }
    
    if (devStatus == 0) {
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

        packetSize = mpu.dmpGetFIFOPacketSize();
    } else {
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
      while(1)
      {
        if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) 
        { 
          #ifdef OUTPUT_ACCEL_GRAVITY
            mpu.dmpGetAccel(&aag, fifoBuffer);
          #endif
          if(counter < sampleFrequency) 
          {
            if(aag.x > 1400) {
              valueCounterX += 1;
            }
            else if(aag.x < -2000) {
              valueCounterY += 1;
            }
            counter++;
          }
          if(counter == sampleFrequency)
          {
            if(valueCounterX > 18 && valueCounterX > valueCounterY) 
            {
              Serial.write('R');
              Serial.write('}');
            }
            else if(valueCounterY > 16 && valueCounterY > valueCounterX) 
            {
              Serial.write('L');
              Serial.write('}');
            }
            else
            {
              Serial.write('N');
              Serial.write('}');
            }
            counter = 0;
            valueCounterX = 0;
            valueCounterY = 0;
          } 
        }
      }
    }
    // blink LED to indicate activity
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
  }   
}
    
