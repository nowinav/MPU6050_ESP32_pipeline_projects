#include <Wire.h>
#include <MPU6050_tockn.h>

// Our proven working pins
#define I2C_SDA 20
#define I2C_SCL 21

// Pass the Wire object to the library
MPU6050 mpu(Wire);

void setup() {
  Serial.begin(115200);
  delay(2000); 

  Serial.println("\n--- MPU6500 Clone Test ---");

  // Start I2C with our pins
  Wire.begin(I2C_SDA, I2C_SCL);

  // Initialize the sensor
  mpu.begin();
  
  // This calculates the gyro offsets. KEEP THE SENSOR STILL for a second.
  Serial.println("Calculating gyro offsets. DO NOT MOVE THE SENSOR...");
  mpu.calcGyroOffsets(true);
  
  Serial.println("Setup complete! Reading data...\n");
  delay(500);
}

void loop() {
  // Update the sensor data
  mpu.update();

  // Print Acceleration
  Serial.print("Accel X: "); Serial.print(mpu.getAccX());
  Serial.print(", Y: ");     Serial.print(mpu.getAccY());
  Serial.print(", Z: ");     Serial.print(mpu.getAccZ());
  Serial.println(" g"); // Note: This library outputs in 'g' (gravity), not m/s^2

  // Print Gyroscope
  Serial.print("Gyro X: ");  Serial.print(mpu.getGyroX());
  Serial.print(", Y: ");     Serial.print(mpu.getGyroY());
  Serial.print(", Z: ");     Serial.print(mpu.getGyroZ());
  Serial.println(" deg/s");

  // Print Temperature
  Serial.print("Temp: ");    Serial.print(mpu.getTemp());
  Serial.println(" degC");

  Serial.println("-----------------------------");
  delay(500);
}