#!/bin/bash

# SignGlove Linux 설정 스크립트
# Linux 사용자를 위한 자동 설정 도구

echo "========================================"
echo "   SignGlove Linux 설정 도구"
echo "========================================"
echo

# 시스템 업데이트
echo "[INFO] 시스템 패키지 업데이트 중..."
sudo apt update

# Python 설치 확인
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3가 설치되지 않았습니다."
    echo "다음 명령어로 Python3를 설치하세요:"
    echo "sudo apt install python3 python3-pip"
    exit 1
fi

echo "[INFO] Python3 설치 확인됨"
python3 --version

# pip 설치 확인
if ! command -v pip3 &> /dev/null; then
    echo "[INFO] pip3 설치 중..."
    sudo apt install python3-pip
fi

# pip 업그레이드
echo "[INFO] pip3 업그레이드 중..."
python3 -m pip install --upgrade pip

# 필수 패키지 설치
echo "[INFO] 필수 패키지 설치 중..."
pip3 install pyserial requests numpy pandas

# 선택적 패키지 설치
echo "[INFO] 선택적 패키지 설치 중..."
pip3 install psutil

# 시리얼 포트 권한 설정
echo "[INFO] 시리얼 포트 권한 설정 중..."
sudo usermod -a -G dialout $USER

# udev 규칙 설정
echo "[INFO] udev 규칙 설정 중..."
UDEV_RULES_FILE="/etc/udev/rules.d/99-arduino.rules"

if [ ! -f "$UDEV_RULES_FILE" ]; then
    echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"2341\", ATTRS{idProduct}==\"*\", MODE=\"0666\"" | sudo tee "$UDEV_RULES_FILE"
    echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"1a86\", ATTRS{idProduct}==\"*\", MODE=\"0666\"" | sudo tee -a "$UDEV_RULES_FILE"
    echo "SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"10c4\", ATTRS{idProduct}==\"*\", MODE=\"0666\"" | sudo tee -a "$UDEV_RULES_FILE"
fi

# udev 규칙 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger

# 사용 가능한 시리얼 포트 확인
echo "[INFO] 사용 가능한 시리얼 포트 확인 중..."
ls /dev/ttyUSB* 2>/dev/null || echo "USB 시리얼 포트가 없습니다."
ls /dev/ttyACM* 2>/dev/null || echo "ACM 시리얼 포트가 없습니다."

echo
echo "========================================"
echo "    설정 완료!"
echo "========================================"
echo
echo "다음 단계를 수행하세요:"
echo "1. 시스템을 재부팅하거나 재로그인하세요"
echo "2. SignGlove_HW를 USB로 연결하세요"
echo "3. 다음 명령어로 시스템을 시작하세요:"
echo "   - API 서버: python3 server/main.py"
echo "   - 통합 클라이언트: python3 integration/signglove_client.py"
echo
echo "문제가 있으면 다음을 확인하세요:"
echo "- 사용자가 dialout 그룹에 추가되었는지: groups \$USER"
echo "- 시리얼 포트 권한: ls -la /dev/ttyUSB*"
echo "- udev 규칙이 적용되었는지: cat $UDEV_RULES_FILE"
echo
