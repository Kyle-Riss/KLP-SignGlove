#!/bin/bash

# SignGlove macOS 설정 스크립트
# macOS 사용자를 위한 자동 설정 도구

echo "========================================"
echo "   SignGlove macOS 설정 도구"
echo "========================================"
echo

# Homebrew 설치 확인
if ! command -v brew &> /dev/null; then
    echo "[INFO] Homebrew 설치 중..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # M1/M2 Mac을 위한 PATH 설정
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

echo "[INFO] Homebrew 설치 확인됨"
brew --version

# Python 설치 확인
if ! command -v python3 &> /dev/null; then
    echo "[INFO] Python3 설치 중..."
    brew install python
fi

echo "[INFO] Python3 설치 확인됨"
python3 --version

# pip 업그레이드
echo "[INFO] pip3 업그레이드 중..."
python3 -m pip install --upgrade pip

# 필수 패키지 설치
echo "[INFO] 필수 패키지 설치 중..."
pip3 install pyserial requests numpy pandas

# 선택적 패키지 설치
echo "[INFO] 선택적 패키지 설치 중..."
pip3 install psutil

# 시리얼 포트 권한 설정 (필요시)
echo "[INFO] 시리얼 포트 권한 확인 중..."
if [ -e "/dev/tty.usbmodem*" ] || [ -e "/dev/tty.usbserial*" ]; then
    echo "[INFO] 시리얼 포트 권한 설정 중..."
    sudo chmod 666 /dev/tty.usbmodem* 2>/dev/null || true
    sudo chmod 666 /dev/tty.usbserial* 2>/dev/null || true
fi

# 사용 가능한 시리얼 포트 확인
echo "[INFO] 사용 가능한 시리얼 포트 확인 중..."
ls /dev/tty.usb* 2>/dev/null || echo "USB 시리얼 포트가 없습니다."
ls /dev/tty.usbserial* 2>/dev/null || echo "USB Serial 포트가 없습니다."
ls /dev/tty.SLAB_USBtoUART* 2>/dev/null || echo "SLAB USB to UART 포트가 없습니다."

# 보안 설정 확인
echo "[INFO] 보안 설정 확인 중..."
echo "macOS 보안 정책으로 인해 시리얼 포트 접근이 차단될 수 있습니다."
echo "시스템 환경설정 → 보안 및 개인 정보 보호에서 확인하세요."

echo
echo "========================================"
echo "    설정 완료!"
echo "========================================"
echo
echo "다음 단계를 수행하세요:"
echo "1. SignGlove_HW를 USB로 연결하세요"
echo "2. 다음 명령어로 시스템을 시작하세요:"
echo "   - API 서버: python3 server/main.py"
echo "   - 통합 클라이언트: python3 integration/signglove_client.py"
echo
echo "문제가 있으면 다음을 확인하세요:"
echo "- 시리얼 포트 권한: ls -la /dev/tty.usb*"
echo "- 보안 정책: 시스템 환경설정 → 보안 및 개인 정보 보호"
echo "- USB 연결: 다른 USB 포트 시도"
echo
echo "추가 도움이 필요하면 다음을 참조하세요:"
echo "- https://pyserial.readthedocs.io/en/latest/pyserial.html"
echo "- https://www.arduino.cc/en/software"
echo
