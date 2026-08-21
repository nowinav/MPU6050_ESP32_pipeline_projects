#include <Wire.h>

#define MPU_ADDR 0x68

#define ENA 9
#define IN1 4
#define IN2 8

#define MOTOR_PWM 128

bool motorRunning = false;

// Sampling frequency
const unsigned long SAMPLE_PERIOD_US = 2000;  // 1 / 500 Hz = 2000 us

unsigned long nextSampleTime = 0;


void setup() {

  Serial.begin(115200);

  Wire.begin();

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // MPU6050 wake up
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission();

  // Motor direction
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);

  // Motor initially OFF
  analogWrite(ENA, 0);

  Serial.println("MPU6050 connected");
  Serial.println("Ready");
}


void loop() {

  // =====================================================
  // CHECK MATLAB COMMAND
  // =====================================================

  if (Serial.available()) {

    char cmd = Serial.read();

    // Motor ON
    if (cmd == '1') {

      motorRunning = true;

      analogWrite(ENA, MOTOR_PWM);

      Serial.println("MOTOR ON");

      // Start sampling schedule
      nextSampleTime = micros();
    }


    // Motor OFF
    if (cmd == '0') {

      motorRunning = false;

      analogWrite(ENA, 0);

      Serial.println("MOTOR OFF");
    }
  }


  // =====================================================
  // 500 Hz SAMPLING
  // =====================================================

  if (motorRunning) {

    unsigned long now = micros();

    if ((long)(now - nextSampleTime) >= 0) {

      nextSampleTime += SAMPLE_PERIOD_US;

      readMPU6050();
    }
  }
}


void readMPU6050() {

  int16_t ax;
  int16_t ay;
  int16_t az;

  // MPU6050 starting register
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, 6, true);

  if (Wire.available() == 6) {

    ax = ((int16_t)Wire.read() << 8) | Wire.read();
    ay = ((int16_t)Wire.read() << 8) | Wire.read();
    az = ((int16_t)Wire.read() << 8) | Wire.read();


    // Send ONLY:
    // AX,AY,AZ

    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.print(",");
    Serial.println(az);
  }
}
