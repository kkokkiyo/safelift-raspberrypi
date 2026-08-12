#include <Wire.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>
#include <SparkFun_MicroPressure.h>

// BNO085 SPI 핀
#define BNO08X_CS      10
#define BNO08X_INT      9
#define BNO08X_RESET    5

// SEN-16476 설정
#define PRESSURE_ADDR  0x18
#define SENSOR_WIRE    Wire1
#define I2C_CLOCK      100000

Adafruit_BNO08x bno08x(BNO08X_RESET);
SparkFun_MicroPressure pressureSensor;
sh2_SensorValue_t sensorValue;

// FSR 6개
const uint8_t fsrPins[6] = {
  A0, A1, A2, A3, A4, A5
};

int fsr[6] = {
  0, 0, 0, 0, 0, 0
};

bool imuAvailable = false;
bool pressureAvailable = false;

// IMU 측정값
float accX = 0.0f;
float accY = 0.0f;
float accZ = 0.0f;

float gyroX = 0.0f;
float gyroY = 0.0f;
float gyroZ = 0.0f;

float quatW = 1.0f;
float quatX = 0.0f;
float quatY = 0.0f;
float quatZ = 0.0f;

// 공기압 측정값
float pressureKPa = 0.0f;

// Unity/Python bridge용 20Hz 출력
const unsigned long outputIntervalMs = 50;
unsigned long lastOutputMs = 0;

void enableImuReports()
{
  if (!bno08x.enableReport(SH2_ACCELEROMETER, 50000)) {
    Serial.println("SLXR_ERROR,Accelerometer enable failed");
  }

  if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 50000)) {
    Serial.println("SLXR_ERROR,Gyroscope enable failed");
  }

  if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 50000)) {
    Serial.println("SLXR_ERROR,Rotation vector enable failed");
  }
}

void setup()
{
  Serial.begin(115200);
  delay(3000);

  Serial.println("SLXR_STATUS,All sensor stream starting");

  analogReadResolution(12);

  SENSOR_WIRE.begin();
  SENSOR_WIRE.setClock(I2C_CLOCK);

  SPI.begin();

  if (bno08x.begin_SPI(BNO08X_CS, BNO08X_INT, &SPI)) {
    imuAvailable = true;
    Serial.println("SLXR_STATUS,BNO085 SPI connected");
    enableImuReports();
  }
  else {
    Serial.println("SLXR_ERROR,BNO085 SPI connection failed");
  }

  if (pressureSensor.begin(PRESSURE_ADDR, SENSOR_WIRE)) {
    pressureAvailable = true;
    Serial.println("SLXR_STATUS,SEN-16476 connected");
  }
  else {
    Serial.println("SLXR_ERROR,SEN-16476 connection failed");
  }

  Serial.println("SLXR_HEADER,millis,fsr1,fsr2,fsr3,fsr4,fsr5,fsr6,accX,accY,accZ,gyroX,gyroY,gyroZ,quatW,quatX,quatY,quatZ,pressure_kPa");
}

void readImu()
{
  if (imuAvailable && bno08x.wasReset()) {
    Serial.println("SLXR_STATUS,BNO085 reset detected");
    enableImuReports();
  }

  if (imuAvailable && bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_ACCELEROMETER:
        accX = sensorValue.un.accelerometer.x;
        accY = sensorValue.un.accelerometer.y;
        accZ = sensorValue.un.accelerometer.z;
        break;

      case SH2_GYROSCOPE_CALIBRATED:
        gyroX = sensorValue.un.gyroscope.x;
        gyroY = sensorValue.un.gyroscope.y;
        gyroZ = sensorValue.un.gyroscope.z;
        break;

      case SH2_ROTATION_VECTOR:
        quatW = sensorValue.un.rotationVector.real;
        quatX = sensorValue.un.rotationVector.i;
        quatY = sensorValue.un.rotationVector.j;
        quatZ = sensorValue.un.rotationVector.k;
        break;
    }
  }
}

void readFsr()
{
  for (int i = 0; i < 6; i++) {
    fsr[i] = analogRead(fsrPins[i]);
  }
}

void readPressure()
{
  if (pressureAvailable) {
    pressureKPa = pressureSensor.readPressure(KPA);
  }
}

void printSlxr()
{
  Serial.print("SLXR,");
  Serial.print(millis());

  for (int i = 0; i < 6; i++) {
    Serial.print(",");
    Serial.print(fsr[i]);
  }

  Serial.print(",");
  Serial.print(accX, 4);

  Serial.print(",");
  Serial.print(accY, 4);

  Serial.print(",");
  Serial.print(accZ, 4);

  Serial.print(",");
  Serial.print(gyroX, 4);

  Serial.print(",");
  Serial.print(gyroY, 4);

  Serial.print(",");
  Serial.print(gyroZ, 4);

  Serial.print(",");
  Serial.print(quatW, 6);

  Serial.print(",");
  Serial.print(quatX, 6);

  Serial.print(",");
  Serial.print(quatY, 6);

  Serial.print(",");
  Serial.print(quatZ, 6);

  Serial.print(",");
  Serial.println(pressureKPa, 3);
}

void loop()
{
  readImu();

  unsigned long now = millis();

  if (now - lastOutputMs < outputIntervalMs) {
    return;
  }

  lastOutputMs = now;

  readFsr();
  readPressure();
  printSlxr();
}