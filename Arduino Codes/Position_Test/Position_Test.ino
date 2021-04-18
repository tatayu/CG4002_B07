//=====================================================
// IMPORTS
//=====================================================
#include <I2Cdev.h>
#include <EEPROM.h>
#include <CircularBuffer.h>
#include "MPU6050_6Axis_MotionApps20.h"

#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  #include "Wire.h"
#endif

//=====================================================
// OPERATING MODES
//=====================================================
// #define DEBUG

//=====================================================
// CONSTANTS
//=====================================================
#define RANGE_ACCEL                         MPU6050_ACCEL_FS_2
#define RANGE_GYRO                          MPU6050_GYRO_FS_250

#define FREQ_LPF                            MPU6050_DLPF_BW_5

// maximum: 200 Hz
#define FREQ_SAMPLING                       200
#define FREQ_PROCESSING                     20
#define FREQ_TRANSMIT                       10
#define FREQ_HEARTBEAT                      1

#define NUM_SAMPLES_CALIBRATION             (FREQ_SAMPLING * 1.00)
#define NUM_SAMPLES_PROCESSING              (FREQ_SAMPLING / FREQ_PROCESSING)

#define SCALE_ACCELERATION                  8

// dynamically assigned
#define THRESHOLD_POS_MECH_FILTER           8
#define THRESHOLD_NEG_MECH_FILTER           -8
#define THRESHOLD_MOVEMENT_START            (int) ceil(FREQ_PROCESSING * 0.50)
#define THRESHOLD_MOVEMENT_END              (int) ceil(FREQ_PROCESSING * 0.50)

#define SIZE_DATA                           7
#define FLAG_CALIBRATE                      'C'
#define FLAG_HEARTBEAT                      'H'
#define FLAG_TIME                           'T'
#define FLAG_CONSTANTS                      'Q'
#define FLAG_DELIMITER_DATA                 '|'
#define FLAG_DELIMITER_TIME                 '#'
#define FLAG_OFFSET                         'o'
#define FLAG_POSITION                       'p'

#define LED_PIN                             13

//=====================================================
// GLOBAL VARIABLES
//=====================================================
MPU6050 mpu;

VectorInt16 a, aa;
// [0]: prev, [1]: curr, [2]: offset
struct VectorInt32 {
  int32_t x;
  int32_t y;
  int32_t z;
} acc[3], vel[2], disp[2];

uint8_t fifoBuffer[64];

struct Position {
  // -1: left, 0: stationary, 1: right
  int p;
  unsigned long m;
};

CircularBuffer<struct Position, 30> transmitBuffer;
CircularBuffer<struct Position, 20> posBuffer;

unsigned long transmitMillis = millis();
unsigned long heartbeatMillis = millis();

int filterThreshold, startThreshold, endThreshold;

//=====================================================
// SETUP
//=====================================================
void setup() {
  #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    Wire.begin();
    Wire.setClock(400000);
  #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
    Fastwire::setup(400, true);
  #endif

  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  setupTimer();

  mpu.initialize();
  if (!mpu.testConnection()) {
    blinkLed(1);
  }

  if (mpu.dmpInitialize()) {
    blinkLed(2);
  }

  if (!readOffsets()) {
    blinkLed(3);
  }

  mpu.setFullScaleAccelRange(RANGE_ACCEL);
  mpu.setFullScaleGyroRange(RANGE_GYRO);

  mpu.setDLPFMode(FREQ_LPF);

  const unsigned char fifoRateDivisor = ((1000.0 / FREQ_SAMPLING) / 5) - 1;
  const unsigned char dmpUpdate[] = {0x00, fifoRateDivisor};
  mpu.writeMemoryBlock(dmpUpdate, 0x02, 0x02, 0x16);

  mpu.setDMPEnabled(true);

  readPositionOffsets();

  readPositionConstants();
}

//=====================================================
// LOOP
//=====================================================
void loop() {
  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    static int cnt = 0;

    getAccel();
    aa.x += a.x;
    aa.y += a.y;
    aa.z += a.z;

    if (++cnt >= NUM_SAMPLES_PROCESSING) {
      acc[1] = { .x = aa.x / cnt, .y = aa.y / cnt, .z = aa.z / cnt };
      memset(&aa, 0, sizeof(aa));
      cnt = 0;

      eliminatePositionOffsets();

      applyMechFilter();

      calculateVelocity();

      calculateDisplacement();

      checkMovement();

      updatePreviousValues();
    }
  }

  if ((millis() - transmitMillis) > (1000.0 / FREQ_TRANSMIT)) {
    struct Position pos = { .p = 0, .m = millis() };
    transmitBuffer.push(pos);
    transmitMillis = millis();
  }

  if (Serial.available()) {
    char c = Serial.read();

    if (c == FLAG_CALIBRATE) {
      calibratePositionOffsets();
    } else if (c == FLAG_TIME) {
      syncTime();
    } else if (c == FLAG_CONSTANTS) {
      setPositionConstants();
    } else if (c == FLAG_HEARTBEAT) {
      if ((millis() - heartbeatMillis) > ((1000.0 / FREQ_HEARTBEAT) + 200.0)) {
        for (int i = 0; i < posBuffer.size(); ++i) {
          for (int j = 0; j < 3; ++j) {
            transmitBuffer.push(posBuffer[i]);
          }
        }
      }

      posBuffer.clear();
      heartbeatMillis = millis();
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    }
  }
}

//=====================================================
// ISR
//=====================================================
ISR(TIMER1_COMPA_vect) {
  transmitData();
}

//=====================================================
// FUNCTIONS
//=====================================================
void setupTimer() {
  cli();

  // clear registers
  TCCR1A = 0;
  TCCR1B = 0;
  TIMSK1 = 0;
  TCNT1  = 0;

  // CTC mode
  TCCR1B |= (1 << WGM12);
  // prescalar = 256
  TCCR1B |= (1 << CS12);
  // enable timer compare interrupt
  TIMSK1 |= (1 << OCIE1A);
  // https://www.arduinoslovakia.eu/application/timer-calculator
  // 20 Hz
  OCR1A = 3124;

  sei();
}

bool readOffsets() {
  int16_t offsets[6];

  Serial.println("Reading IMU offsets in EEPROM...");

  int i = 0;
  if (EEPROM.read(i++) == FLAG_OFFSET) {
    Serial.println("IMU offsets");
    Serial.println("===========");
    for (int j = 0; j < 6; ++j) {
      offsets[j] = (EEPROM.read(i++) << 8) + EEPROM.read(i++);
      Serial.print(offsets[j]);
      Serial.print(" ");
    }
    Serial.println("");

    mpu.setXAccelOffset(offsets[0]);
    mpu.setYAccelOffset(offsets[1]);
    mpu.setZAccelOffset(offsets[2]);
    mpu.setXGyroOffset(offsets[3]);
    mpu.setYGyroOffset(offsets[4]);
    mpu.setZGyroOffset(offsets[5]);

    return true;
  } else {
    Serial.println("IMU offsets not found in EEPROM...");
    return false;
  }
}

void calibratePositionOffsets() {
  #ifdef DEBUG
    Serial.println("Calibrating offsets...");
  #endif

  delay(1000);

  VectorInt16 aaa = { .x = 0, .y = 0, .z = 0 };
  int cnt = 0;
  while (cnt++ < NUM_SAMPLES_CALIBRATION) {
    while (!mpu.dmpGetCurrentFIFOPacket(fifoBuffer));
    getAccel();
    aaa.x += a.x;
    aaa.y += a.y;
    aaa.z += a.z;
  }

  int i = 13;
  EEPROM.write(i++, FLAG_POSITION);
  EEPROM.write(i++, aaa.x / cnt);
  EEPROM.write(i++, aaa.y / cnt);
  EEPROM.write(i++, aaa.z / cnt);

  #ifdef DEBUG
    Serial.println("Calibrated offsets...");
  #endif

  readPositionOffsets();
}

void readPositionOffsets() {
  int i = 13;
  if (EEPROM.read(i++) == FLAG_POSITION) {
    acc[2].x = (int8_t) EEPROM.read(i++);
    acc[2].y = (int8_t) EEPROM.read(i++);
    acc[2].z = (int8_t) EEPROM.read(i++);
  }

  #ifdef DEBUG
    Serial.println("Position offsets");
    Serial.println("================");
    Serial.print("[");
    Serial.print(acc[2].x);
    Serial.print(" ");
    Serial.print(acc[2].y);
    Serial.print(" ");
    Serial.print(acc[2].z);
    Serial.println("]");
  #endif
}

void setPositionConstants() {
  #ifdef DEBUG
    Serial.println("Assigning constants...");
  #endif

  int i = 17;
  EEPROM.write(i++, FLAG_CONSTANTS);
  // filter, start, end
  for (int j = 0; j < 3; ++j) {
    while (!Serial.available());
    EEPROM.write(i++, int(Serial.read()));
  }

  #ifdef DEBUG
    Serial.println("Assigned constants...");
  #endif
}

void readPositionConstants() {
  int i = 17;
  if (EEPROM.read(i++) == FLAG_CONSTANTS) {
    filterThreshold = EEPROM.read(i++);
    startThreshold = ceil(FREQ_PROCESSING * ((EEPROM.read(i++) * 2) / 100.0));
    endThreshold = ceil(FREQ_PROCESSING * ((EEPROM.read(i++) * 2) / 100.0));
  }

  #ifdef DEBUG
    Serial.println("Position constants");
    Serial.println("==================");
    Serial.print("[");
    Serial.print(filterThreshold);
    Serial.print(" ");
    Serial.print(startThreshold);
    Serial.print(" ");
    Serial.print(endThreshold);
    Serial.println("]");
  #endif
}

void syncTime() {
  const int timeout = 1000;
  uint8_t buf[7];
  uint8_t state = 0;

  digitalWrite(LED_PIN, HIGH);

  while (true) {
    // flush
    while (Serial.available() && Serial.read());

    unsigned long ts = millis();
    buf[0] = FLAG_DELIMITER_TIME;
    buf[1] = state;
    buf[2] = ts >> 24;
    buf[3] = ts >> 16;
    buf[4] = ts >> 8;
    buf[5] = ts;
    buf[6] = calculateChecksum(buf, sizeof(buf));

    Serial.write(buf, sizeof(buf));
    while (!Serial.available() && ((millis() - ts) < timeout));

    char c = Serial.read();
    if (c == 'Y') {
      state = 1;
    } else if (c == 'Z') {
      break;
    }
  }

  digitalWrite(LED_PIN, LOW);
}

void getAccel() {
  mpu.dmpGetAccel(&a, fifoBuffer);

  a.x >>= SCALE_ACCELERATION;
  a.y >>= SCALE_ACCELERATION;
  a.z >>= SCALE_ACCELERATION;
}

void eliminatePositionOffsets() {
  acc[1].x -= acc[2].x;
  acc[1].y -= acc[2].y;
  acc[1].z -= acc[2].z;
}

void applyMechFilter() {
  if (acc[1].x >= THRESHOLD_NEG_MECH_FILTER && acc[1].x <= THRESHOLD_POS_MECH_FILTER) {
    acc[1].x = 0;
  }

  if (acc[1].y >= THRESHOLD_NEG_MECH_FILTER && acc[1].y <= THRESHOLD_POS_MECH_FILTER) {
    acc[1].y = 0;
  }

  if (acc[1].z >= THRESHOLD_NEG_MECH_FILTER && acc[1].z <= THRESHOLD_POS_MECH_FILTER) {
    acc[1].z = 0;
  }
}

void calculateVelocity() {
  vel[1].x = vel[0].x + acc[0].x + ((acc[1].x - acc[0].x) >> 1);
  vel[1].y = vel[0].y + acc[0].y + ((acc[1].y - acc[0].y) >> 1);
  vel[1].z = vel[0].z + acc[0].z + ((acc[1].z - acc[0].z) >> 1);
}

void calculateDisplacement() {
  disp[1].x = disp[0].x + vel[0].x + ((vel[1].x - vel[0].x) >> 1);
  disp[1].y = disp[0].y + vel[0].y + ((vel[1].y - vel[0].y) >> 1);
  disp[1].z = disp[0].z + vel[0].z + ((vel[1].z - vel[0].z) >> 1);
}

void updatePreviousValues() {
  acc[0] = acc[1];
  vel[0] = vel[1];
  disp[0] = disp[1];
}

void checkMovement() {
  static int moveCnt[3], endCnt[3], dir;

  // update endCnt
  endCnt[0] = (acc[1].x == 0) ? (endCnt[0] + 1) : 0;
  endCnt[1] = (acc[1].y == 0) ? (endCnt[1] + 1) : 0;
  endCnt[2] = (acc[1].z == 0) ? (endCnt[2] + 1) : 0;

  // update moveCnt
  for (int i = 0; i < 3; ++i) {
    moveCnt[i] = (endCnt[i] < THRESHOLD_MOVEMENT_END) ? (moveCnt[i] + 1) : 0;
  }

  // tmp
  #ifdef DEBUG
    static int prevMoveCnt;
    if (moveCnt[1] == 0 && prevMoveCnt > 0) {
      Serial.print(dir);
      Serial.print(" ");
      Serial.println(prevMoveCnt);
    }
    prevMoveCnt = moveCnt[1];
  #endif

  // check for valid movement
  dir = (moveCnt[1] == 1) ? sgn(disp[1].y) : (moveCnt[1] == 0) ? 0 : dir;
  if (moveCnt[1] == THRESHOLD_MOVEMENT_START) {
    struct Position pos = { .p = dir, .m = millis() };
    for (int i = 0; i < 3; ++i) {
      transmitBuffer.push(pos);
    }
    posBuffer.push(pos);
  }

  // check if movement has ended
  if (endCnt[0] >= THRESHOLD_MOVEMENT_END) {
    vel[1].x = 0;
    disp[1].x = 0;
  }
  if (endCnt[1] >= THRESHOLD_MOVEMENT_END) {
    vel[1].y = 0;
    disp[1].y = 0;
  }
  if (endCnt[2] >= THRESHOLD_MOVEMENT_END) {
    vel[1].z = 0;
    disp[1].z = 0;
  }
}

void transmitData() {
  if (transmitBuffer.isEmpty()) {
    return;
  }

  int i = 0;
  uint8_t dataBuffer[SIZE_DATA];
  struct Position pos = transmitBuffer.shift();

  // delimiter, millis, pos, checksum
  dataBuffer[i++] = FLAG_DELIMITER_DATA;
  dataBuffer[i++] = pos.m >> 24;
  dataBuffer[i++] = pos.m >> 16;
  dataBuffer[i++] = pos.m >> 8;
  dataBuffer[i++] = pos.m;
  dataBuffer[i++] = pos.p;
  dataBuffer[i++] = calculateChecksum(dataBuffer, SIZE_DATA);

  #ifndef DEBUG
    Serial.write(dataBuffer, sizeof(dataBuffer));
  #endif
}

uint8_t calculateChecksum(uint8_t buf[], int sz) {
  uint8_t checksum = 0;

  // omit delimiter and checksum fields
  for (int i = 1; i < (sz - 1); ++i) {
    checksum ^= buf[i];
  }

  return checksum;
}

void blinkLed(int num) {
  const int blinkInterval = 250;
  const int offInterval = 1000;

  while (true) {
    for (int i = 0; i < num; ++i) {
      digitalWrite(LED_PIN, HIGH);
      delay(blinkInterval);
      digitalWrite(LED_PIN, LOW);
      delay(blinkInterval);
    }

    digitalWrite(LED_PIN, LOW);
    delay(offInterval);
  }
}

int sgn(int val) {
  return (val > 0) - (val < 0);
}
