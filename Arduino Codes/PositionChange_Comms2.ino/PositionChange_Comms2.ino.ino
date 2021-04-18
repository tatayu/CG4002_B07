// I2C device class (I2Cdev) demonstration Arduino sketch for MPU6050 class using DMP (MotionApps v2.0)
// 6/21/2012 by Jeff Rowberg <jeff@rowberg.net>
// Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
//
// Changelog:
//      2019-07-08 - Added Auto Calibration and offset generator
//       - and altered FIFO retrieval sequence to avoid using blocking code
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
int valueCounterZ = 0;
int16_t aaXPrevious = 0;
int16_t aaYPrevious = 0;
int16_t aaZPrevious = 0;
int16_t aaXDiff = 0;
int16_t aaYDiff = 0;
int16_t aaZDiff = 0;
int16_t aaXTotal = 0;
int16_t aaYTotal = 0;
int16_t aaZTotal = 0;

void setOffSet(int x_gyro, int y_gyro, int z_gyro, int z_accel) {
  mpu.setXGyroOffset(x_gyro);
  mpu.setYGyroOffset(y_gyro);
  mpu.setZGyroOffset(z_gyro);
  mpu.setZAccelOffset(z_accel); // 1688 factory default for my test chip
}

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

    switch(bluno){
      case 1 : 
        setOffSet(216, -40, 33, 597); // Arm Beetle 1
        break;
      case 2 : 
        setOffSet(253,-103, 8, 1804); // Arm Beetle 2
        break;
      case 3 : 
        setOffSet(81, -25,-34, 2122); // Arm Beetle 3
        break;
      case 4 : 
        setOffSet(155,-37, -56, 1006); // Waist Beetle 1
        break;
      case 5 :
        setOffSet(254, -55, 18, 973); // Waist Beetle 2
        break;
      case 6 : 
        setOffSet(80, 36, -19, 1770); //Waist Beetle 3 
        break;
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
            if (valueCounterX == 0 && valueCounterY == 0) {
              if(aag.x > 3000) {
                valueCounterX = 1;
              }
              else if(aag.x < -3000) {
                valueCounterY = 1;
              }  
            }
            if (aag.y > 5000) {
              valueCounterZ += 1;
            }
            counter++;
          }
          if(counter == sampleFrequency)
          {
            if(valueCounterZ > 2) {
              Serial.write('E');
              Serial.write('}');
            }
            else if(valueCounterX == 1) //Beetle3w: 15, 1400 Beetle2w: 14 1100 Beetle1w: 16 1550
            {
              Serial.write('R');
              Serial.write('}');
            }
            else if(valueCounterY == 1) //Beetle3w: 16, -1500 Beetle2w: 17 -1500  Beetle1w: 14 -1300
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
            valueCounterZ = 0;
          } 
        }
      }
    }
    // blink LED to indicate activity
    blinkState = !blinkState;
    digitalWrite(LED_PIN, blinkState);
  }   
}
    
