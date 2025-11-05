import os
import sys
import time
import json
import signal
from collections import deque
from typing import Optional, List

import requests

try:
    import serial
except ImportError as e:
    print("pyserial이 필요합니다. 설치: pip install pyserial", file=sys.stderr)
    raise


API_URL = os.environ.get("SIGNGLOVE_API_URL", "http://localhost:8000/predict")
SERIAL_PORT = os.environ.get("SIGNGLOVE_SERIAL_PORT", None)  # 예: /dev/ttyACM0
BAUDRATE = int(os.environ.get("SIGNGLOVE_BAUDRATE", "115200"))
WINDOW = int(os.environ.get("SIGNGLOVE_WINDOW", "87"))
CHANNELS = int(os.environ.get("SIGNGLOVE_CHANNELS", "8"))
TOP_K = int(os.environ.get("SIGNGLOVE_TOPK", "5"))
NORMALIZE = os.environ.get("SIGNGLOVE_NORMALIZE", "true").lower() != "false"


def parse_line(line: str) -> Optional[List[float]]:
    """
    장갑에서 전송되는 한 줄을 파싱하여 길이 CHANNELS의 float 배열로 변환
    예상 포맷 예: "ts, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7"
    타임스탬프 컬럼이 있을 수 있으므로 뒤 CHANNELS개만 사용
    """
    try:
        parts = [p.strip() for p in line.strip().split(',') if p.strip() != '']
        if len(parts) < CHANNELS:
            return None
        values = list(map(float, parts[-CHANNELS:]))
        if len(values) != CHANNELS:
            return None
        return values
    except Exception:
        return None


def main() -> int:
    if SERIAL_PORT is None:
        print("환경변수 SIGNGLOVE_SERIAL_PORT 를 설정하세요. 예: /dev/ttyACM0", file=sys.stderr)
        return 2

    print(f"[bridge] Open serial {SERIAL_PORT} @ {BAUDRATE}")
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    buf: deque = deque(maxlen=WINDOW)

    def handle_sigint(sig, frame):
        print("\n[bridge] terminating...")
        try:
            ser.close()
        finally:
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        line = ser.readline().decode('utf-8', errors='ignore')
        if not line:
            continue
        row = parse_line(line)
        if row is None:
            continue
        buf.append(row)
        if len(buf) == WINDOW:
            payload = {
                "data": list(buf),
                "top_k": TOP_K,
                "normalize": NORMALIZE,
            }
            try:
                resp = requests.post(API_URL, json=payload, timeout=2)
                if resp.ok:
                    out = resp.json()
                    top1 = out.get("predicted_class")
                    conf = None
                    tks = out.get("top_k_predictions") or []
                    if tks:
                        conf = tks[0].get("probability")
                    print(f"[pred] {top1} {f'({conf:.2f})' if conf is not None else ''}")
                else:
                    print(f"[pred] HTTP {resp.status_code}")
            except Exception as e:
                print(f"[pred] error: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


