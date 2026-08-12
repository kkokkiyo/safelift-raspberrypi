# Sensor Protocol

Future serial sensors can send newline-delimited JSON:

```json
{
  "timestamp": 12345.1,
  "left_pressure_ratio": 0.52,
  "right_pressure_ratio": 0.48,
  "heel_pressure_drop": false,
  "forefoot_bias": false,
  "forefoot_shift_percent": 75.0,
  "fsr_scoring_type": "A",
  "imu_pitch": 8.5,
  "imu_roll": 2.1,
  "imu_instability": 0.12,
  "core_pressure_ratio": 0.71
}
```

Run with a serial provider:

```powershell
python main.py --camera auto --serial_port COM3
```

Run with a JSONL replay provider:

```powershell
python main.py --camera auto --sensor_jsonl Backend/docs/sample_sensor_frames.jsonl
```

From the repository root:

```powershell
.\run_backend.ps1 -SensorJsonl .\Backend\docs\sample_sensor_frames.jsonl -OpenHealth
```

Run with an HTTP push provider:

```powershell
.\run_backend.ps1 -SensorHttp -OpenHealth
```

Then a bridge process can POST each already-normalized frame:

```powershell
$body = @{
  left_pressure_ratio = 0.52
  right_pressure_ratio = 0.48
  core_pressure_ratio = 0.91
  imu_pitch = 4.0
  imu_roll = 1.0
  fsr_scoring_type = "A"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8090/api/sensor_frame -Body $body -ContentType "application/json"
```

`core_pressure_ratio` is optional, but when present it is used for the rule-based `CorePressureDrop` live feedback cue and report evidence. It should be normalized to 0.0-1.0 before it reaches this backend contract.


## Backend contract validation

The repository does not implement raw Arduino/SafeLift acquisition. It expects
the connected provider to publish already-normalized values. The backend now
validates that boundary before Unity Real Mode treats a sensor as ready:

- `left_pressure_ratio`, `right_pressure_ratio`, `core_pressure_ratio`, and
  `imu_instability` must be finite ratios in `0.0..1.0`.
- `imu_pitch` and `imu_roll` must be finite degree values in `-180.0..180.0`.
- `heel_pressure_drop` and `forefoot_bias` must be boolean-compatible.
- `fsr_scoring_type` is optional and must be `A` or `B`.
  - `A`: 일반 사용자 채점. 좌우 불균형 23점 + 뒤꿈치 유지 12점.
  - `B`: 가동범위 제한/보정 채점. 좌우 불균형 23점 + 앞꿈치 안정성 12점.
- `forefoot_shift_percent` is optional and is used for Type B scoring. It represents the live forefoot pressure increase over the calibrated forefoot baseline. Type B grants the 12 forefoot-stability points up to 100%, then subtracts 1.2 points per excess 10%.
- empty serial lines, invalid JSON, non-object JSON, or out-of-range values are
  published as `contract_ok=false` with `contract_errors`, and `/api/health`
  reports the sensor provider as not ready.

The JSONL replay provider and HTTP push provider use the exact same validation
boundary as serial. They are useful when Arduino/SafeLift raw acquisition is not
ready yet: the hardware or bridge team can export or POST normalized frames
first, then Unity Real Mode can be checked against deterministic or live sensor
evidence.

This keeps the sensor raw acquisition out of scope while making the normalized
provider contract strict enough for Quest/Unity to run without silently using
bad values.
