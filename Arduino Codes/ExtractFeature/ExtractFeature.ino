/*
  Example of use of the FFT libray to compute FFT for several signals over a range of frequencies.
        The exponent is calculated once before the excecution since it is a constant.
        This saves resources during the excecution of the sketch and reduces the compiled size.
        The sketch shows the time that the computing is taking.
        Copyright (C) 2014 Enrique Condes
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "arduinoFFT.h"

arduinoFFT FFT = arduinoFFT(); /* Create FFT object */
/*
These values can be changed in order to evaluate the functions
*/

const uint16_t samples = 128; //This value MUST ALWAYS be a power of 2
const double samplingFrequency = 100.0; //Hz, must be less than 10000 due to ADC


/*
These are the input and output vectors
Input vectors receive computed results from FFT
*/
double frequency[samples/2];
double vReal[samples];
double vImag[samples];

void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(115200);
}

void loop() {
  double totalVoltage = 0;
  double voltageMeanSquare = 0;
  float voltageRootMeanSquare = 0;
  float meanAmplitude = 0;
  long long powerSpectrum = 0;
  long long powerSpectrumFrequency = 0;
  double meanFreq = 0;

  for(int i=0; i< samples; i++) 
  {
    double sensorValue = analogRead(A0);
    //Serial.println(sensorValue);
    float voltage = sensorValue * (5.0 / 1023.0); 

    vReal[i] = sensorValue;
    vImag[i] = 0;
    
    voltageMeanSquare += (voltage * voltage); //V^2

    totalVoltage += voltage;

    if(i < samples / 2) {
      frequency[i] = ((i * samplingFrequency) / double(samples));
    }
    
  }

  FFT.Windowing(vReal, samples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(vReal, vImag, samples, FFT_FORWARD);
  FFT.ComplexToMagnitude(vReal, vImag, samples);
  
  voltageMeanSquare = voltageMeanSquare / (samples*1.0); // VMS/sampleSize
  voltageRootMeanSquare = sqrt(voltageMeanSquare); // RMS = square root of sum of squared voltage / sample size

  meanAmplitude = totalVoltage / float(samples);


  for(int i=0; i<samples; i++) {
    double power = vReal[i] * vReal[i];
    powerSpectrum += power;
    powerSpectrumFrequency += powerSpectrum * frequency[i];
  }

  meanFreq = double(powerSpectrumFrequency) / double(powerSpectrum); 
  
  
  Serial.println("MNF: ");
  Serial.println(meanFreq);
  Serial.print("Voltage RMS: ");
  Serial.println(voltageRootMeanSquare);
  Serial.println("MAV: ");
  Serial.println(meanAmplitude);
  
  

  delay(5000);
  

  
  
}
