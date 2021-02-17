uint16_t accel_1 = 100;
uint16_t accel_2 = 200;
uint16_t accel_3 = 300;

uint16_t gyro_1 = 400;
uint16_t gyro_2 = 500;
uint16_t gyro_3 = 60000;

uint16_t accel[] = {accel_1, accel_2, accel_3};
bool start_flag = false;

void setup() {
  Serial.begin(115200);
  
}

void loop() {
  if(Serial.available())
  { 
    //handshake
    char packet_type = Serial.read();
    if(packet_type == 'H')
    {
      Serial.write('A');
    }
    //start transmission of data
    else if (packet_type == 'A')
    {
      Serial.write('{');
      Serial.write('I');
      Serial.write((const char *) &accel_1, sizeof(accel_1));
      Serial.write((const char *) &accel_2, sizeof(accel_2));
      Serial.write((const char *) &accel_3, sizeof(accel_3));
      Serial.write((const char *) &gyro_1, sizeof(gyro_1));
      Serial.write((const char *) &gyro_2, sizeof(gyro_2));
      Serial.write((const char *) &gyro_3, sizeof(gyro_3));
      Serial.write('}');
    }
  }
}
