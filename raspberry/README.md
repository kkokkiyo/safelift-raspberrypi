# SafeLift Raspberry Pi

Arduino 없이 Raspberry Pi가 센서를 직접 읽고, OpenCV/MediaPipe 카메라 분석과 기존 SafeLift 평가 로직을 결합하는 실행용 프로젝트입니다.

## 하드웨어 기본 구성

- FSR 6개: 전압분배회로 → MCP3008 CH0~CH5 → SPI
- IMU: BNO085 SPI (기존 Arduino 스케치와 동일한 센서)
- 공기압 센서: SEN-16476/MPRLS I2C 0x18 (기존 Arduino 스케치와 동일한 센서)
- 웹캠: USB V4L2 카메라 또는 Raspberry Pi Camera를 OpenCV가 열 수 있는 장치

Raspberry Pi GPIO에는 일반 아날로그 입력이 없으므로 FSR을 GPIO에 직접 연결하지 않습니다. SPI/I2C는 `sudo raspi-config`에서 활성화합니다.

BNO085 기본 핀은 `CS=D8`, `INT=D25`, `RESET=D24`로 설정했습니다. 실제 배선이 다르면 `hardware/imu_reader.py`의 핀 인자를 수정합니다.

## PC에서 빠른 테스트

```powershell
cd D:\embedded\raspberry
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py --mock --no-camera
```

브라우저: `http://127.0.0.1:8090`

통합 스모크 테스트:

```bash
python smoke_test.py
```

`SMOKE_TEST_PASS`가 나오면 하드웨어 없이 센서→캘리브레이션→점수→위험도→웹 API 연결이 정상입니다. 실제 하드웨어 판정은 반드시 Raspberry Pi에서 각 센서를 별도로 확인한 뒤 진행합니다.

## Raspberry Pi 실행

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv i2c-tools
sudo raspi-config   # SPI, I2C 활성화
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py --camera 0 --host 0.0.0.0
```

처음에는 `--mock`로 웹 화면과 평가 파이프라인을 확인한 뒤, `--camera 0`과 실제 MCP3008/IMU/공기압 드라이버를 연결합니다.

## API

- `GET /api/state`: 최신 센서·카메라·점수·위험도
- `POST /api/calibration/start`: 3초 캘리브레이션 시작
- `POST /api/session/start`: 세션 시작
- `POST /api/session/stop`: 세션 종료
- `GET /api/health`: 장치 상태

카메라나 센서가 실패해도 웹 서버는 종료되지 않고 `/api/health`와 `/api/state`에 오류를 표시합니다.

## 현재 평가 로직

기존 `score_rubric.py`의 기준을 Raspberry Pi용으로 옮겼습니다.

- Core/공기압: 35점
- FSR 좌우 균형: 35점
- MediaPipe 몸통각과 IMU pitch 일치도: 30점

센서값은 먼저 캘리브레이션하고, 카메라/센서 중 하나의 신뢰도가 낮으면 해당 항목을 `partial`로 표시합니다. 실제 대회용 임계값은 실측 데이터로 조정해야 합니다.
