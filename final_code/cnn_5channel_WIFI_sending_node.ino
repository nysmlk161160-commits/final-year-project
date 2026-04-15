#include <SPI.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include "I2Cdev.h"
#include "MPU6050.h"
#include <math.h>
#include <stdint.h>

const char* SSID = "Glide_Resident";
const char* PASS = "ScootJulyClash";

IPAddress PYTHON_IP(10, 133, 138, 31);
const uint16_t PYTHON_PORT = 5005;

const uint16_t LOCAL_UDP_PORT = 4001;

MPU6050 mpu;
int16_t ax, ay, az;
int16_t gx, gy, gz;

typedef struct {
  float angle;
  float bias;
  float P[2][2];
} KalmanState;

KalmanState kalmanRoll  = {0, 0, {{1, 0}, {0, 1}}};
KalmanState kalmanPitch = {0, 0, {{1, 0}, {0, 1}}};

typedef struct {
  uint32_t t_ms;
  uint16_t a0;
  uint16_t a1;
  uint16_t a2;
  uint16_t a3;
  uint16_t a4;
  float roll;
  float pitch;
} SensorFrame;

typedef struct __attribute__((packed)) {
  uint32_t t_ms;
  uint16_t a0;
  uint16_t a1;
  uint16_t a2;
  uint16_t a3;
  uint16_t a4;
  int16_t roll_x100;
  int16_t pitch_x100;
} BinaryFrame;

typedef struct __attribute__((packed)) {
  uint16_t magic;
  uint16_t version;
  uint32_t batch_id;
  BinaryFrame frames[5];
} BatchPacket;

WiFiUDP udp;

const unsigned long SAMPLE_INTERVAL_MS = 20;
const uint8_t BATCH_SIZE = 5;
unsigned long lastSampleTime = 0;
unsigned long lastKalmanTime = 0;

SensorFrame frameBuf[BATCH_SIZE];
uint8_t frameCount = 0;
uint32_t batchId = 0;

unsigned long lastDebugPrintMs = 0;
const unsigned long DEBUG_PRINT_INTERVAL_MS = 1000;

float kalmanUpdate(KalmanState &k, float accAngle, float gyroRate, float dt) {
  const float Q_angle = 0.001f;
  const float Q_bias  = 0.003f;
  const float R_meas  = 0.03f;

  float rate = gyroRate - k.bias;
  k.angle += dt * rate;

  k.P[0][0] += dt * (dt * k.P[1][1] - k.P[0][1] - k.P[1][0] + Q_angle);
  k.P[0][1] -= dt * k.P[1][1];
  k.P[1][0] -= dt * k.P[1][1];
  k.P[1][1] += Q_bias * dt;

  float y = accAngle - k.angle;
  float S = k.P[0][0] + R_meas;
  float K0 = k.P[0][0] / S;
  float K1 = k.P[1][0] / S;

  k.angle += K0 * y;
  k.bias  += K1 * y;

  float P00 = k.P[0][0];
  float P01 = k.P[0][1];

  k.P[0][0] -= K0 * P00;
  k.P[0][1] -= K0 * P01;
  k.P[1][0] -= K1 * P00;
  k.P[1][1] -= K1 * P01;

  return k.angle;
}

void connectWiFi() {
  Serial.print("Connecting to WiFi SSID: ");
  Serial.println(SSID);

  while (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(SSID, PASS);

    unsigned long t0 = millis();
    while (millis() - t0 < 10000) {
      if (WiFi.status() == WL_CONNECTED) {
        break;
      }
      delay(500);
      Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi connect failed, retrying...");
      delay(2000);
    }
  }

  Serial.println("WiFi connected.");
  Serial.print("Local IP: ");
  Serial.println(WiFi.localIP());
}

int16_t scaleAngle100(float x) {
  long v = lroundf(x * 100.0f);
  if (v > 32767) v = 32767;
  if (v < -32768) v = -32768;
  return (int16_t)v;
}

BinaryFrame makeBinaryFrame(const SensorFrame& f) {
  BinaryFrame bf;
  bf.t_ms = f.t_ms;
  bf.a0 = f.a0;
  bf.a1 = f.a1;
  bf.a2 = f.a2;
  bf.a3 = f.a3;
  bf.a4 = f.a4;
  bf.roll_x100 = scaleAngle100(f.roll);
  bf.pitch_x100 = scaleAngle100(f.pitch);
  return bf;
}

bool readOneSensorFrame(SensorFrame& outFrame) {
  unsigned long now = millis();

  // Read flex sensors
  uint16_t a0 = (uint16_t)analogRead(A0);
  uint16_t a1 = (uint16_t)analogRead(A1);
  uint16_t a2 = (uint16_t)analogRead(A2);
  uint16_t a3 = (uint16_t)analogRead(A3);
  uint16_t a4 = (uint16_t)analogRead(A4);

  // Read IMU data
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  float dt = (now - lastKalmanTime) / 1000.0f;
  if (dt <= 0.0f) dt = 0.02f;
  lastKalmanTime = now;

  float accRoll  = atan2((float)ay, (float)az) * 180.0f / PI;
  float accPitch = atan2(-(float)ax, sqrt((float)ay * ay + (float)az * az)) * 180.0f / PI;

  float gyroRateRoll  = gx / 131.0f;
  float gyroRatePitch = gy / 131.0f;

  float roll  = kalmanUpdate(kalmanRoll,  accRoll,  gyroRateRoll,  dt);
  float pitch = kalmanUpdate(kalmanPitch, accPitch, gyroRatePitch, dt);

  outFrame.t_ms = (uint32_t)now;
  outFrame.a0 = a0;
  outFrame.a1 = a1;
  outFrame.a2 = a2;
  outFrame.a3 = a3;
  outFrame.a4 = a4;
  outFrame.roll = roll;
  outFrame.pitch = pitch;

  return true;
}

bool sendBatchPacket(const SensorFrame* frames, uint32_t thisBatchId) {
  BatchPacket pkt;
  pkt.magic = 0x4753;
  pkt.version = 1;
  pkt.batch_id = thisBatchId;

  for (uint8_t i = 0; i < BATCH_SIZE; i++) {
    pkt.frames[i] = makeBinaryFrame(frames[i]);
  }

  int beginOk = udp.beginPacket(PYTHON_IP, PYTHON_PORT);
  if (!beginOk) {
    Serial.println("udp.beginPacket() failed");
    return false;
  }

  size_t n = udp.write((const uint8_t*)&pkt, sizeof(BatchPacket));
  int endOk = udp.endPacket();

  if (n != sizeof(BatchPacket) || endOk == 0) {
    Serial.print("UDP send failed, bytes=");
    Serial.print(n);
    Serial.print(", expected=");
    Serial.println(sizeof(BatchPacket));
    return false;
  }

  return true;
}

void setup() {
  Serial.begin(500000);
  delay(1500);

  Serial.println("=== Arduino1 WiFi UDP Sensor Sender Start ===");

  // MKR boards can support higher ADC resolution if needed
  // analogReadResolution(12);

  Wire.begin();
  Wire.setClock(400000);
  mpu.initialize();

  delay(1000);

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed!");
  } else {
    Serial.println("MPU6050 connected.");
  }

  if (sizeof(BinaryFrame) != 18) {
    Serial.print("ERROR: BinaryFrame size = ");
    Serial.println(sizeof(BinaryFrame));
    while (1) {}
  }

  if (sizeof(BatchPacket) != 98) {
    Serial.print("ERROR: BatchPacket size = ");
    Serial.println(sizeof(BatchPacket));
    while (1) {}
  }

  connectWiFi();

  if (!udp.begin(LOCAL_UDP_PORT)) {
    Serial.println("UDP begin failed!");
    while (1) {}
  }

  lastSampleTime = millis();
  lastKalmanTime = millis();

  Serial.print("UDP local port: ");
  Serial.println(LOCAL_UDP_PORT);
  Serial.print("Python target: ");
  Serial.print(PYTHON_IP);
  Serial.print(":");
  Serial.println(PYTHON_PORT);

  Serial.println("Sampling: 20 ms per frame");
  Serial.println("Send mode: one UDP packet for every 5 frames");
  Serial.print("BinaryFrame bytes = ");
  Serial.println(sizeof(BinaryFrame));
  Serial.print("BatchPacket bytes = ");
  Serial.println(sizeof(BatchPacket));
}

void loop() {
  unsigned long now = millis();

  // Reconnect automatically if WiFi drops
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting...");
    connectWiFi();
  }

  // Sample at a fixed 20 ms interval
  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime += SAMPLE_INTERVAL_MS;

    // Prevent long-term drift after occasional blocking
    if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
      lastSampleTime = now;
    }

    SensorFrame f;
    if (readOneSensorFrame(f)) {
      frameBuf[frameCount] = f;
      frameCount++;

      // Send one batch once 5 frames have been collected
      if (frameCount >= BATCH_SIZE) {
        bool ok = sendBatchPacket(frameBuf, batchId);
        if (ok) {
          batchId++;
        }
        frameCount = 0;
      }
    }
  }

  // Print debug info at a low rate
  if (now - lastDebugPrintMs >= DEBUG_PRINT_INTERVAL_MS) {
    lastDebugPrintMs = now;
    Serial.print("WiFi status=");
    Serial.print(WiFi.status());
    Serial.print(", batchId=");
    Serial.println(batchId);
  }
}
