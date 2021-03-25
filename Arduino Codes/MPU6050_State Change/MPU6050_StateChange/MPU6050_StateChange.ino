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

// I2Cdev and MPU6050 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"

#include "MPU6050_6Axis_MotionApps20.h"
//#include "MPU6050.h" // not necessary if using MotionApps include file

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
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
// ===                      INITIAL SETUP                       ===
// ================================================================

bool isMoving = true;
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
    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
        Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    // initialize serial communication
    // (115200 chosen because it is required for Teapot Demo output, but it's
    // really up to you depending on your project)
    Serial.begin(115200);
    while (!Serial); // wait for Leonardo enumeration, others continue immediately

    // NOTE: 8MHz or slower host processors, like the Teensy @ 3.3V or Arduino
    // Pro Mini running at 3.3V, cannot handle this baud rate reliably due to
    // the baud timing being too misaligned with processor ticks. You must use
    // 38400 or slower in these cases, or use some kind of external separate
    // crystal solution for the UART timer.

    // initialize device
    Serial.println(F("Initializing I2C devices..."));
    mpu.initialize();
    pinMode(INTERRUPT_PIN, INPUT);

    // verify connection
    Serial.println(F("Testing device connections..."));
    Serial.println(mpu.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));

    // wait for ready
    Serial.println(F("\nSend any character to begin DMP programming and demo: "));
    while (Serial.available() && Serial.read()); // empty buffer
    while (!Serial.available());                 // wait for data
    while (Serial.available() && Serial.read()); // empty buffer again

    // load and configure the DMP
    Serial.println(F("Initializing DMP..."));
    devStatus = mpu.dmpInitialize();
    
    
    int bluno = 1;
    // supply your own gyro offsets here, scaled for min sensitivity
    
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
      mpu.setXGyroOffset(259);
      mpu.setYGyroOffset(-97);
      mpu.setZGyroOffset(19);
      mpu.setZAccelOffset(3578);
    }
    else if(bluno == 5)
    {
      mpu.setXGyroOffset(259);
      mpu.setYGyroOffset(-97);
      mpu.setZGyroOffset(19);
      mpu.setZAccelOffset(3578);
    }
    else if(bluno == 6)
    {
      mpu.setXGyroOffset(259);
      mpu.setYGyroOffset(-97);
      mpu.setZGyroOffset(19);
      mpu.setZAccelOffset(3578);
    }
    else if(bluno == 7)
    {
      mpu.setXGyroOffset(259);
      mpu.setYGyroOffset(-97);
      mpu.setZGyroOffset(19);
      mpu.setZAccelOffset(3578);
    }
    
      
    // make sure it worked (returns 0 if so)
    if (devStatus == 0) {
        // Calibration Time: generate offsets and calibrate our MPU6050
        mpu.CalibrateAccel(6);
        mpu.CalibrateGyro(6);
        mpu.PrintActiveOffsets();
        // turn on the DMP, now that it's ready
        Serial.println(F("Enabling DMP..."));
        mpu.setDMPEnabled(true);

        // enable Arduino interrupt detection
        Serial.print(F("Enabling interrupt detection (Arduino external interrupt "));
        Serial.print(digitalPinToInterrupt(INTERRUPT_PIN));
        Serial.println(F(")..."));
        attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
        mpuIntStatus = mpu.getIntStatus();

        // set our DMP Ready flag so the main loop() function knows it's okay to use it
        Serial.println(F("DMP ready! Waiting for first interrupt..."));
        dmpReady = true;

        // get expected DMP packet size for later comparison
        packetSize = mpu.dmpGetFIFOPacketSize();
    } else {
        // ERROR!
        // 1 = initial memory load failed
        // 2 = DMP configuration updates failed
        // (if it's going to break, usually the code will be 1)
        Serial.print(F("DMP Initialization failed (code "));
        Serial.print(devStatus);
        Serial.println(F(")"));
    }

    // configure LED for output
    pinMode(LED_PIN, OUTPUT);
}



// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================

void loop() {
    // if programming failed, don't try to do anything
    if (!dmpReady) return;
    // read a packet from FIFO
    if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) { // Get the Latest packet 

        #ifdef OUTPUT_ACCEL_GRAVITY
            // display acceleration with gravity included
            mpu.dmpGetAccel(&aaGravity, fifoBuffer); // Reads from the packet, raw accel data
//            Serial.print("acc\t");
//            Serial.print(aaGravity.x);
//            Serial.print("\t");
//            Serial.print(aaGravity.y);
//            Serial.print("\t");
//            Serial.println(aaGravity.z);
        #endif*/

        if(counter == 0) 
        {
          aaXPrevious = aaGravity.x;
          aaYPrevious = aaGravity.y;
          aaZPrevious = aaGravity.z;

          counter++;
        } else if(counter > 0 && counter < sampleFrequency)
        {
          aaXDiff = abs(aaGravity.x - aaXPrevious);
          aaXPrevious = aaGravity.x;
          
          aaYDiff = abs(aaGravity.y - aaYPrevious);
          aaYPrevious = aaGravity.y;
          
          aaZDiff = abs(aaGravity.z - aaZPrevious);
          aaZPrevious = aaGravity.z;

          
          aaXTotal += aaXDiff;
          aaYTotal += aaYDiff;
          aaZTotal += aaZDiff;

          counter++;
        } else if(counter == sampleFrequency) 
        {
          Serial.print("Acceleration Diffrence Total\t");
          Serial.print(aaXTotal);
          Serial.print("\t");
          Serial.print(aaYTotal);
          Serial.print("\t");
          Serial.print(aaZTotal);

        if(aaXTotal > 150 && aaXTotal < 1000 && aaYTotal > 145 && aaYTotal < 700 && aaZTotal > 200 && aaZTotal < 700) {          
          isMoving = false;
          Serial.println("Dancer is not moving");
        } else 
        {
          isMoving = true;
          Serial.println("Dancer is moving");
        }

        counter = 0;
        aaXPrevious = 0;
        aaYPrevious = 0;
        aaZPrevious = 0;
        aaXDiff = 0;
        aaYDiff = 0;
        aaZDiff = 0;
        aaXTotal = 0;
        aaYTotal = 0;
        aaZTotal = 0;
    }
        
        // blink LED to indicate activity
        blinkState = !blinkState;
        digitalWrite(LED_PIN, blinkState);
    }
}