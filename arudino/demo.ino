#include <Arduino.h>

#define SENSOR_PIN 15

volatile unsigned long pulses = 0;

portMUX_TYPE pulseMux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR countPulse()
{
    portENTER_CRITICAL_ISR(&pulseMux);
    pulses++;
    portEXIT_CRITICAL_ISR(&pulseMux);
}

void setup()
{
    Serial.begin(115200);
    delay(500);

    pinMode(SENSOR_PIN, INPUT);

    attachInterrupt(
        digitalPinToInterrupt(SENSOR_PIN),
        countPulse,
        RISING
    );

    Serial.println("Eddy Current Oscillator Frequency Measurement");
    Serial.println("Waiting for oscillator signal...");
}

void loop()
{
    delay(1000);

    portENTER_CRITICAL(&pulseMux);
    unsigned long count = pulses;
    pulses = 0;
    portEXIT_CRITICAL(&pulseMux);

    Serial.print("Frequency: ");
    Serial.print(count);
    Serial.println(" Hz");
}