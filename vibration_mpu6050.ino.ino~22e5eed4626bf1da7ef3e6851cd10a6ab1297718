#include <Wire.h>

// MPU6050 I2C Address (0x68 if AD0 is connected to GND)
const int MPU_ADDR = 0x68; 

// Variables to hold the raw data
int16_t raw_acc_x, raw_acc_y, raw_acc_z;
int16_t raw_gyro_x, raw_gyro_y, raw_gyro_z;
int16_t raw_temp;

// Variables for calculated values
float acc_x, acc_y, acc_z;       // in g
float gyro_x, gyro_y, gyro_z;    // in degrees/sec
float temperature;               // in Celsius

void setup() {
  Serial.begin(115200);
  
  // Initialize I2C on ESP32-CAM pins (SDA=13, SCL=14)
  Wire.begin(13, 14);
  Wire.setClock(400000); // Fast mode I2C for higher sampling rate
  
  // --- Wake up the MPU6050 ---
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0x00);  // Set to 0 to wake up the sensor
  Wire.endTransmission(true);
  
  // --- Configure Accelerometer Range (±8g) ---
  // Standard range is ±2g (16384 LSB/g). ±8g is 4096 LSB/g, better for vibrations.
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1C);  // ACCEL_CONFIG register
  Wire.write(0x10);  // 0x10 = ±8g range
  Wire.endTransmission(true);

  // --- Configure Gyroscope Range (±500 deg/s) ---
  // Standard range is ±250 (131 LSB/dps). ±500 is 65.5 LSB/dps.
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x1B);  // GYRO_CONFIG register
  Wire.write(0x08);  // 0x08 = ±500 deg/s range
  Wire.endTransmission(true);

  // Print CSV Header
  Serial.println("AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Temp");
  
  delay(1000); // Wait for sensor to stabilize
}

void loop() {
  // --- Read all 14 bytes starting from register 0x3B ---
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); 
  Wire.endTransmission(false);
  // Request 14 bytes: 6 Accel, 2 Temp, 6 Gyro
  Wire.requestFrom(MPU_ADDR, 14, true); 

  if (Wire.available() == 14) {
    // 1. Read Accelerometer (6 bytes)
    raw_acc_x = (Wire.read() << 8) | Wire.read();
    raw_acc_y = (Wire.read() << 8) | Wire.read();
    raw_acc_z = (Wire.read() << 8) | Wire.read();
    
    // 2. Read Temperature (2 bytes)
    raw_temp = (Wire.read() << 8) | Wire.read();
    
    // 3. Read Gyroscope (6 bytes)
    raw_gyro_x = (Wire.read() << 8) | Wire.read();
    raw_gyro_y = (Wire.read() << 8) | Wire.read();
    raw_gyro_z = (Wire.read() << 8) | Wire.read();

    // --- Convert raw data to physical units ---
    
    // Accel: ±8g range means divide by 4096.0
    acc_x = raw_acc_x / 4096.0;
    acc_y = raw_acc_y / 4096.0;
    acc_z = raw_acc_z / 4096.0;

    // Temperature: Formula from datasheet
    temperature = (raw_temp / 340.0) + 36.53;

    // Gyro: ±500 deg/s range means divide by 65.5
    gyro_x = raw_gyro_x / 65.5;
    gyro_y = raw_gyro_y / 65.5;
    gyro_z = raw_gyro_z / 65.5;

    // --- Print to Serial Monitor in CSV format ---
    Serial.print(acc_x, 3); Serial.print(",");
    Serial.print(acc_y, 3); Serial.print(",");
    Serial.print(acc_z, 3); Serial.print(",");
    
    Serial.print(gyro_x, 2); Serial.print(",");
    Serial.print(gyro_y, 2); Serial.print(",");
    Serial.print(gyro_z, 2); Serial.print(",");
    
    Serial.println(temperature, 2);
  }

  // Sampling Delay. 
  // 10ms = ~100Hz sampling rate. 
  // Change to 5ms for ~200Hz (better for high-frequency vibration).
  delay(10); 
}