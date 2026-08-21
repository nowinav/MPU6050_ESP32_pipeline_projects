#include <Wire.h>

#define MPU_ADDR 0x68

#define ENA 9
#define IN1 4
#define IN2 8

#define MOTOR_PWM 128
boolean Switch=true;
void setup() {
  
  Serial.begin(115200);

  Wire.begin();

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission();

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  analogWrite(ENA, MOTOR_PWM);

  Serial.println("Motor running at 50% PWM");
  Serial.println("MPU6050 connected");
}

void loop() {

  // Check for command
if (Serial.available()) {

  char cmd = Serial.read();

  if (cmd == '1') {
    Switch = true;
    analogWrite(ENA, MOTOR_PWM);
    Serial.println("MOTOR ON");
  }

  if (cmd == '0') {
    Switch = false;
    analogWrite(ENA, 0);
    Serial.println("MOTOR OFF");
  }
}

  // Only read MPU when system is ON
  if (Switch == true) {

    int16_t ax, ay, az;
    int16_t temp;
    int16_t gx, gy, gz;

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    Wire.endTransmission(false);

    Wire.requestFrom(MPU_ADDR, 14, true);

    if (Wire.available() == 14) {

      ax = Wire.read() << 8 | Wire.read();
      ay = Wire.read() << 8 | Wire.read();
      az = Wire.read() << 8 | Wire.read();

      temp = Wire.read() << 8 | Wire.read();

      gx = Wire.read() << 8 | Wire.read();
      gy = Wire.read() << 8 | Wire.read();
      gz = Wire.read() << 8 | Wire.read();

      Serial.print(ax);
      Serial.print(",");
      Serial.print(ay);
      Serial.print(",");
      Serial.println(az);
    }

    delay(20);
  }
}
