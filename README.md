# SafeLift Embedded

> 이 저장소는 [SafeLift XR](https://kkokkiyo.github.io/projects/safelift-xr/)(2026 GIST AI창의융합경진대회 최우수상)을 Unity와 아두이노 없이 라즈베리파이 하나로 옮겨 보려던 실험입니다. 실제 대회에 쓴 버전이 아니고, 끝까지 완성하지 않았습니다.

Raspberry Pi에서 FSR·IMU·공기압 센서와 웹캠을 직접 읽고 OpenCV·MediaPipe·규칙 기반 평가 로직으로 운동 자세와 위험도를 분석하는 프로젝트입니다. Unity와 Arduino 없이 Raspberry Pi 하나에서 센서 수집, 계산, 웹 대시보드까지 실행하는 것을 목표로 합니다.

## 구성

```text
embedded/
├── raspberry/          # Raspberry Pi 실행 프로젝트
├── arduino/            # 기존 센서 펌웨어·웹캠 보조 참고 자료
└── final/Backend/      # 기존 Python 분석·리포트 참고 자료
```

실제 Raspberry Pi 실행은 `raspberry/`에서 시작합니다.

## 기능

- MCP3008 SPI를 통한 FSR 6채널 입력
- BNO085 IMU 쿼터니언 기반 pitch/roll/yaw 계산
- SEN-16476/MPRLS 공기압 센서 I2C 입력
- USB 웹캠 OpenCV 입력
- MediaPipe Pose 관절·무릎각·몸통각 추정
- 센서 캘리브레이션, 점수, 위험도, Flask 웹 대시보드
- 하드웨어 없는 Mock 모드와 자동 테스트

## 하드웨어

- Raspberry Pi 5 8GB, Raspberry Pi OS 64-bit 권장
- MCP3008 ADC + FSR 6개 + 센서별 전압분배회로
- BNO085 IMU
- SEN-16476/MPRLS 공기압 센서
- USB 웹캠 또는 Raspberry Pi Camera

Raspberry Pi 일반 GPIO에는 아날로그 입력이 없습니다. FSR을 GPIO에 직접 연결하지 말고 MCP3008을 사용하세요. GPIO는 3.3V 기준이므로 5V 신호를 직접 입력하면 안 됩니다.

기본 MCP3008 연결:

```text
CLK  → GPIO11 / SPI SCLK
DOUT → GPIO9  / SPI MISO
DIN  → GPIO10 / SPI MOSI
CS   → GPIO8  / SPI CE0
FSR 6개 → CH0~CH5
VDD/VREF → 3.3V, AGND/DGND → GND
```

## 설치

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv i2c-tools v4l-utils
sudo raspi-config
```

`raspi-config`에서 SPI와 I2C를 활성화합니다.

```bash
i2cdetect -y 1
v4l2-ctl --list-devices
rpicam-hello --list-cameras
cd ~/embedded/raspberry
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행

하드웨어 없는 Mock 테스트:

```bash
cd ~/embedded/raspberry
source .venv/bin/activate
python smoke_test.py
python app.py --mock --no-camera --host 0.0.0.0
```

실제 웹캠 포함:

```bash
python app.py --camera 0 --host 0.0.0.0 --port 8090
```

브라우저 주소: `http://<RASPBERRY_PI_IP>:8090`

브라우저는 `/api/state`를 0.5초마다 읽고 `/camera.mjpg` MJPEG 스트림으로 영상을 표시합니다.

## API

```text
GET  /api/state
GET  /api/health
POST /api/calibration/start
POST /api/session/start
POST /api/session/stop
GET  /camera.mjpg
```

## 테스트

```bash
python -m py_compile app.py smoke_test.py hardware/*.py analysis/*.py vision/*.py tests/*.py
python -m unittest discover -s tests -v
python smoke_test.py
```

`SMOKE_TEST_PASS`가 출력되면 Mock 센서→캘리브레이션→점수→위험도→API 흐름이 통과한 것입니다.

## 실제 측정

1. 센서·카메라를 연결합니다.
2. `/api/health`에서 장치 상태를 확인합니다.
3. 브라우저에서 캘리브레이션을 시작하고 3초간 기본 자세를 유지합니다.
4. 측정을 시작해 MediaPipe 관절, 센서값, 점수, 위험도를 확인합니다.

## 평가 기준

```text
공기압/코어 브레이싱       35점
FSR 좌우 하중 균형         35점
MediaPipe 몸통각-IMU pitch 30점
```

실제 대회 전에는 사용자별 실측 데이터로 임계값을 검증해야 합니다. 의료 진단 장치가 아닌 운동 안전 보조용 프로토타입입니다.

## systemd 자동 실행

```ini
[Unit]
Description=SafeLift Raspberry Pi Web App
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/embedded/raspberry
ExecStart=/home/pi/embedded/raspberry/.venv/bin/python app.py --camera 0 --host 0.0.0.0 --port 8090
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now safelift
```

실제 BNO085/MPRLS 배선·주소와 Raspberry Pi OS/Python 버전에 따라 드라이버 조정이 필요할 수 있습니다. 센서나 카메라가 초기화되지 않아도 웹 서버는 오류 상태를 표시하도록 설계되어 있습니다.
