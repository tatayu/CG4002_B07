void setup() 
{
  Serial.begin(9600);
}
 
void loop() 
{
  float sensorValue = analogRead(A3);
  float millivolt = (sensorValue/1023)*5;
  
  Serial.print("Sensor Value: ");
  Serial.println(sensorValue);
  
  Serial.print("Voltage: ");
  Serial.print(millivolt*1000);
  Serial.println(" mV");
  Serial.println("");
  delay(1);       
}
