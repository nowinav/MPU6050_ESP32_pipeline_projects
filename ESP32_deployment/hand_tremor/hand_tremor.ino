// Adaptive tremor-tracking notch filter
// ESP32-CAM (AI Thinker) + MPU6050, SDA=GPIO13, SCL=GPIO14
// Pipeline: 100 Hz sampling -> Goertzel scan 3-12 Hz every 0.5 s -> retune biquad notch -> CSV out
// CSV: t_ms,ax,ay,az,fx,fy,fz,f0_hz,peak_ratio,loop_us,gx,gy,gz

#include <Wire.h>
#include <math.h>

#define SDA_PIN 13
#define SCL_PIN 14
#define MPU_ADDR 0x68

const float FS = 100.0f;            // sampling rate (Hz)
const uint32_t TS_US = 10000;       // sample period (us)
const int N = 200;                  // analysis window = 2 s
const int HOP = 50;                 // re-estimate every 0.5 s
const float F_MIN = 3.0f, F_MAX = 12.0f, F_STEP = 0.5f;
const float NOTCH_Q = 4.0f;         // higher Q = narrower notch
const float DETECT_RATIO = 3.0f;    // peak/mean power needed to accept a tremor

struct Biquad {
  float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
  float z1 = 0, z2 = 0;
  float process(float x) {          // transposed direct form II
    float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
  void setNotch(float f0) {
    float w0 = 2.0f * PI * f0 / FS;
    float alpha = sinf(w0) / (2.0f * NOTCH_Q);
    float c = cosf(w0);
    float a0 = 1.0f + alpha;
    b0 = 1.0f / a0;
    b1 = -2.0f * c / a0;
    b2 = 1.0f / a0;
    a1 = -2.0f * c / a0;
    a2 = (1.0f - alpha) / a0;
  }
};

Biquad nf[3];                       // one notch per accel axis
float buf[3][N];                    // ring buffer of raw accel (g)
int head = 0, count = 0, sinceEst = 0;
float f0 = 0.0f;                    // tracked tremor frequency (0 = none yet)
float lastRatio = 0.0f;

void mpuInit() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);

  Wire.beginTransmission(MPU_ADDR);   // wake up
  Wire.write(0x6B); Wire.write(0x00);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);   // accel +/-8g
  Wire.write(0x1C); Wire.write(0x10);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);   // gyro +/-500 dps
  Wire.write(0x1B); Wire.write(0x08);
  Wire.endTransmission(true);

  Wire.beginTransmission(MPU_ADDR);   // DLPF ~44 Hz (anti-alias for 100 Hz sampling)
  Wire.write(0x1A); Wire.write(0x03);
  Wire.endTransmission(true);
}

bool readMPU(float *a, float *g) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14, (uint8_t)true) != 14) return false;

  int16_t r[7];
  for (int i = 0; i < 7; i++) r[i] = (Wire.read() << 8) | Wire.read();
  // r[0..2]=accel, r[3]=temp, r[4..6]=gyro
  for (int i = 0; i < 3; i++) a[i] = r[i] / 4096.0f;       // g
  for (int i = 0; i < 3; i++) g[i] = r[4 + i] / 65.5f;     // deg/s
  return true;
}

// Goertzel power for one axis at frequency f (Hann window, mean removed)
float goertzelPower(int axis, float f, float mean) {
  float k = f * N / FS;
  float coeff = 2.0f * cosf(2.0f * PI * k / N);
  float s1 = 0, s2 = 0;
  for (int i = 0; i < N; i++) {
    int idx = (head + i) % N;       // oldest -> newest
    float w = 0.5f - 0.5f * cosf(2.0f * PI * i / (N - 1));
    float s = (buf[axis][idx] - mean) * w + coeff * s1 - s2;
    s2 = s1; s1 = s;
  }
  return s1 * s1 + s2 * s2 - coeff * s1 * s2;
}

void estimateTremor() {
  float mean[3] = {0, 0, 0};
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < N; i++) mean[a] += buf[a][i];
    mean[a] /= N;
  }
  float best = 0, sum = 0, bestF = 0;
  int bins = 0;
  for (float f = F_MIN; f <= F_MAX + 1e-3f; f += F_STEP) {
    float p = 0;
    for (int a = 0; a < 3; a++) p += goertzelPower(a, f, mean[a]);
    sum += p; bins++;
    if (p > best) { best = p; bestF = f; }
  }
  float avg = sum / bins;
  lastRatio = (avg > 0) ? best / avg : 0;
  if (lastRatio > DETECT_RATIO) {
    f0 = (f0 == 0.0f) ? bestF : 0.8f * f0 + 0.2f * bestF;  // smooth estimate
    for (int a = 0; a < 3; a++) nf[a].setNotch(f0);
  }
}

void setup() {
  Serial.begin(230400);
  mpuInit();
  delay(1000);
  Serial.println("t_ms,ax,ay,az,fx,fy,fz,f0_hz,peak_ratio,loop_us,gx,gy,gz");
}

void loop() {
  static uint32_t next = micros();
  while ((int32_t)(micros() - next) < 0) { }   // wait for exact 10 ms tick
  next += TS_US;

  uint32_t t0 = micros();
  float a[3], g[3];
  if (!readMPU(a, g)) return;

  for (int i = 0; i < 3; i++) buf[i][head] = a[i];
  head = (head + 1) % N;
  if (count < N) count++;

  float y[3];
  bool active = (f0 > 0.0f);
  for (int i = 0; i < 3; i++) y[i] = active ? nf[i].process(a[i]) : a[i];

  if (count == N && ++sinceEst >= HOP) { sinceEst = 0; estimateTremor(); }

  uint32_t dt = micros() - t0;
  Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%lu,%.2f,%.2f,%.2f\n",
                millis(), a[0], a[1], a[2], y[0], y[1], y[2],
                f0, lastRatio, dt, g[0], g[1], g[2]);
}