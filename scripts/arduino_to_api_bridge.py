import os
import sys
import time
import json
import signal
from pathlib import Path
from collections import deque
from typing import Optional, List

import requests


"""
SignGlove 레포의 arduino_interface.py를 직접 사용하여
실시간 센서 데이터를 FastAPI /predict로 스트리밍하는 브리지.

환경변수:
  SIGNGLOVE_REPO_SRC   : /home/billy/25-1kp/SignGlove/src (필수)
  SIGNGLOVE_API_URL    : http://localhost:8000/predict (기본)
  SIGNGLOVE_WINDOW     : 87 (기본)
  SIGNGLOVE_CHANNELS   : 8  (기본)
  SIGNGLOVE_TOPK       : 5  (기본)
  SIGNGLOVE_NORMALIZE  : true/false (기본: true)

실행 예:
  export SIGNGLOVE_REPO_SRC=/home/billy/25-1kp/SignGlove/src
  python3 scripts/arduino_to_api_bridge.py
"""


API_URL = os.environ.get("SIGNGLOVE_API_URL", "http://localhost:8000/predict")
REPO_SRC = os.environ.get("SIGNGLOVE_REPO_SRC")
WINDOW = int(os.environ.get("SIGNGLOVE_WINDOW", "87"))
CHANNELS = int(os.environ.get("SIGNGLOVE_CHANNELS", "8"))
TOP_K = int(os.environ.get("SIGNGLOVE_TOPK", "5"))
NORMALIZE = os.environ.get("SIGNGLOVE_NORMALIZE", "true").lower() != "false"


if not REPO_SRC:
    print("환경변수 SIGNGLOVE_REPO_SRC 를 설정하세요. 예: /home/billy/25-1kp/SignGlove/src", file=sys.stderr)
    sys.exit(2)

if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

try:
    from arduino_interface import SignGloveArduinoInterface, ArduinoConfig
except Exception as e:
    print(f"SignGlove arduino_interface import 실패: {e}", file=sys.stderr)
    sys.exit(3)


def parse_line_to_vector(line: str) -> Optional[List[float]]:
    # 예상 포맷: "ts, ch0, ch1, ..., ch7" 혹은 "ch0, ch1, ..., ch7"
    try:
        parts = [p.strip() for p in line.strip().split(',') if p.strip() != '']
        if len(parts) < CHANNELS:
            return None
        vals = list(map(float, parts[-CHANNELS:]))
        if len(vals) != CHANNELS:
            return None
        return vals
    except Exception:
        return None


def main() -> int:
    buf: deque = deque(maxlen=WINDOW)
    last_sent_ts = 0.0
    logs_dir = Path(__file__).resolve().parent.parent / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = logs_dir / 'raw_stream.csv'
    window_csv = logs_dir / 'raw_windows.csv'

    config = ArduinoConfig(
        port=None,                # 자동 탐지
        baudrate=115200,
        auto_detect=True,
        auto_reconnect=True,
    )
    arduino = SignGloveArduinoInterface(config)

    def on_connected(_):
        print("[bridge] Arduino connected")

    def on_disconnected(_):
        print("[bridge] Arduino disconnected")

    def on_error(data):
        print(f"[bridge] Arduino error: {data}")

    def on_data_received(reading):
        # reading.raw_line 가 있으면 사용, 없으면 구성
        raw = getattr(reading, 'raw_line', None)
        if raw is None:
            # fallback: 숫자 필드 조합 시도
            # reading 가 속성 ch0..ch7 을 가진다면 이를 사용
            fields = []
            for i in range(CHANNELS):
                val = getattr(reading, f'ch{i}', None)
                if val is None:
                    return
                fields.append(val)
            vec = [float(v) for v in fields]
        else:
            vec = parse_line_to_vector(raw)
            if vec is None:
                return

        buf.append(vec)
        # append raw line to CSV: ts, ch0..ch7
        try:
            ts = time.time()
            with open(raw_csv, 'a', encoding='utf-8') as f:
                f.write(str(ts))
                for v in vec:
                    f.write(',')
                    f.write(str(v))
                f.write('\n')
        except Exception:
            pass
        if len(buf) == WINDOW:
            payload = {
                "data": list(buf),
                "top_k": TOP_K,
                "normalize": NORMALIZE,
            }
            # save current window snapshot (first row only for brevity mark start)
            try:
                with open(window_csv, 'a', encoding='utf-8') as f:
                    f.write('# window_start ' + str(time.time()) + '\n')
            except Exception:
                pass
            try:
                resp = requests.post(API_URL, json=payload, timeout=2)
                if resp.ok:
                    out = resp.json()
                    top1 = out.get("predicted_class")
                    prob = None
                    tks = out.get("top_k_predictions") or []
                    if tks:
                        prob = tks[0].get("probability")
                    print(f"[pred] {top1} {f'({prob:.2f})' if prob is not None else ''}")
                else:
                    print(f"[pred] HTTP {resp.status_code}")
            except Exception as e:
                print(f"[pred] error: {e}")

    arduino.register_callback('on_connected', on_connected)
    arduino.register_callback('on_disconnected', on_disconnected)
    arduino.register_callback('on_error', on_error)
    arduino.register_callback('on_data_received', on_data_received)

    def handle_sigint(sig, frame):
        print("\n[bridge] terminating...")
        try:
            arduino.disconnect()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    if not arduino.connect():
        print("[bridge] 초기 연결 실패. 자동재연결 대기 중...")

    # 콜백 기반이므로 sleep 만 유지
    while True:
        time.sleep(0.1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


