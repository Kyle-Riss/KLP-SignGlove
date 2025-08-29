# SignGlove 크로스 플랫폼 설정 가이드

**Windows, macOS, Linux에서 SignGlove 통합 시스템 사용하기**

## 🖥️ 지원 플랫폼

| 플랫폼 | 버전 | 아키텍처 | 상태 |
|--------|------|----------|------|
| **Windows** | 10/11 | x64 | ✅ 완전 지원 |
| **macOS** | 10.15+ | Intel/Apple Silicon | ✅ 완전 지원 |
| **Linux** | Ubuntu 18.04+ | x64/ARM | ✅ 완전 지원 |

## 🚀 빠른 시작

### Windows 사용자
```cmd
# 1. 자동 설정 스크립트 실행
integration\windows_setup.bat

# 2. API 서버 시작
python server/main.py

# 3. 통합 클라이언트 실행
python integration/signglove_client.py
```

### macOS 사용자
```bash
# 1. 자동 설정 스크립트 실행
./integration/macos_setup.sh

# 2. API 서버 시작
python3 server/main.py

# 3. 통합 클라이언트 실행
python3 integration/signglove_client.py
```

### Linux 사용자
```bash
# 1. 자동 설정 스크립트 실행
./integration/linux_setup.sh

# 2. 시스템 재부팅 또는 재로그인

# 3. API 서버 시작
python3 server/main.py

# 4. 통합 클라이언트 실행
python3 integration/signglove_client.py
```

## 📋 상세 설정 가이드

### Windows 설정

#### 1. 시스템 요구사항
- **OS**: Windows 10 (버전 1903+) 또는 Windows 11
- **Python**: 3.8 이상
- **메모리**: 최소 4GB RAM
- **저장공간**: 최소 1GB 여유 공간

#### 2. Python 설치
1. [Python.org](https://www.python.org/downloads/)에서 최신 버전 다운로드
2. 설치 시 **"Add Python to PATH"** 옵션 체크
3. 설치 완료 후 명령 프롬프트에서 확인:
   ```cmd
   python --version
   ```

#### 3. Arduino 드라이버 설치
1. [Arduino IDE](https://www.arduino.cc/en/software) 다운로드
2. 설치 시 드라이버 포함 옵션 선택
3. SignGlove_HW를 USB로 연결
4. 장치 관리자에서 포트 확인

#### 4. 포트 확인 방법
```cmd
# PowerShell에서
Get-WmiObject -Class Win32_SerialPort | Select-Object Name, DeviceID, Description

# 또는 장치 관리자에서
# 포트(COM & LPT) → Arduino Uno (COM3)
```

### macOS 설정

#### 1. 시스템 요구사항
- **OS**: macOS 10.15 (Catalina) 이상
- **Python**: 3.8 이상
- **아키텍처**: Intel 또는 Apple Silicon (M1/M2)

#### 2. Homebrew 설치
```bash
# Homebrew 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# M1/M2 Mac PATH 설정
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile
```

#### 3. Python 설치
```bash
# Homebrew로 Python 설치
brew install python

# 버전 확인
python3 --version
```

#### 4. 시리얼 포트 권한 설정
```bash
# 포트 권한 설정
sudo chmod 666 /dev/tty.usbmodem*
sudo chmod 666 /dev/tty.usbserial*

# 포트 확인
ls /dev/tty.usb*
```

#### 5. 보안 정책 설정
1. 시스템 환경설정 → 보안 및 개인 정보 보호
2. 개인 정보 보호 → 완전한 디스크 접근 권한
3. 터미널 앱에 권한 부여

### Linux 설정

#### 1. 시스템 요구사항
- **OS**: Ubuntu 18.04+, CentOS 7+, Debian 9+
- **Python**: 3.8 이상
- **아키텍처**: x64 또는 ARM

#### 2. 시스템 패키지 업데이트
```bash
sudo apt update
sudo apt upgrade
```

#### 3. Python 설치
```bash
# Python3 및 pip 설치
sudo apt install python3 python3-pip

# 버전 확인
python3 --version
pip3 --version
```

#### 4. 시리얼 포트 권한 설정
```bash
# 사용자를 dialout 그룹에 추가
sudo usermod -a -G dialout $USER

# udev 규칙 설정
sudo nano /etc/udev/rules.d/99-arduino.rules

# 다음 내용 추가:
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="*", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="*", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="*", MODE="0666"

# udev 규칙 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger
```

#### 5. 시스템 재부팅
```bash
# 변경사항 적용을 위해 재부팅
sudo reboot
```

## 🔧 플랫폼별 특수 설정

### Windows 특수 설정

#### COM 포트 충돌 해결
```cmd
# 장치 관리자에서 포트 번호 변경
# 포트(COM & LPT) → Arduino Uno → 속성 → 포트 설정 → 고급 → COM 포트 번호
```

#### 드라이버 문제 해결
```cmd
# 장치 관리자에서 드라이버 업데이트
# 또는 Arduino IDE 재설치
```

#### 권한 문제 해결
```cmd
# 관리자 권한으로 명령 프롬프트 실행
# 또는 PowerShell을 관리자 권한으로 실행
```

### macOS 특수 설정

#### USB 연결 불안정 해결
```bash
# 다른 USB 포트 시도
# 또는 USB 허브 사용

# 시스템 정보에서 USB 확인
system_profiler SPUSBDataType
```

#### 보안 정책 우회
```bash
# 터미널에 완전한 디스크 접근 권한 부여
# 시스템 환경설정 → 보안 및 개인 정보 보호 → 개인 정보 보호
```

#### M1/M2 Mac 최적화
```bash
# Rosetta 2 설치 (필요시)
softwareupdate --install-rosetta

# ARM 네이티브 Python 사용 권장
brew install python
```

### Linux 특수 설정

#### udev 규칙 문제 해결
```bash
# udev 규칙 확인
cat /etc/udev/rules.d/99-arduino.rules

# udev 로그 확인
sudo udevadm monitor --property

# 수동으로 포트 권한 설정
sudo chmod 666 /dev/ttyUSB0
```

#### 사용자 그룹 문제 해결
```bash
# 현재 사용자 그룹 확인
groups $USER

# dialout 그룹에 추가
sudo usermod -a -G dialout $USER

# 새 그룹 적용을 위해 재로그인
newgrp dialout
```

#### 커널 모듈 문제 해결
```bash
# USB 시리얼 모듈 로드
sudo modprobe usbserial
sudo modprobe ch341
sudo modprobe cp210x

# 부팅 시 자동 로드
echo "usbserial" | sudo tee -a /etc/modules
echo "ch341" | sudo tee -a /etc/modules
echo "cp210x" | sudo tee -a /etc/modules
```

## 🧪 플랫폼별 테스트

### Windows 테스트
```cmd
# 플랫폼 유틸리티 테스트
python integration/platform_utils.py

# 통합 시스템 테스트
python integration/test_integration.py

# 포트 연결 테스트
python -c "import serial; print([p.device for p in serial.tools.list_ports.comports()])"
```

### macOS 테스트
```bash
# 플랫폼 유틸리티 테스트
python3 integration/platform_utils.py

# 통합 시스템 테스트
python3 integration/test_integration.py

# 포트 연결 테스트
python3 -c "import serial; print([p.device for p in serial.tools.list_ports.comports()])"
```

### Linux 테스트
```bash
# 플랫폼 유틸리티 테스트
python3 integration/platform_utils.py

# 통합 시스템 테스트
python3 integration/test_integration.py

# 포트 연결 테스트
python3 -c "import serial; print([p.device for p in serial.tools.list_ports.comports()])"
```

## 🔍 문제 해결

### 공통 문제

#### 1. Python 버전 문제
```bash
# Python 버전 확인
python --version  # Windows
python3 --version  # macOS/Linux

# 가상환경 사용 권장
python -m venv signglove_env
source signglove_env/bin/activate  # macOS/Linux
signglove_env\Scripts\activate     # Windows
```

#### 2. 패키지 설치 문제
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 클리어
pip cache purge

# 강제 재설치
pip install --force-reinstall pyserial requests numpy pandas
```

#### 3. 권한 문제
```bash
# 사용자 설치
pip install --user pyserial requests numpy pandas

# 또는 가상환경 사용
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 플랫폼별 문제

#### Windows 특수 문제
- **COM 포트 인식 안됨**: 장치 관리자에서 드라이버 업데이트
- **권한 오류**: 관리자 권한으로 실행
- **Python PATH 문제**: 환경 변수 확인

#### macOS 특수 문제
- **보안 정책 차단**: 시스템 환경설정에서 권한 부여
- **USB 연결 불안정**: 다른 USB 포트 시도
- **M1/M2 호환성**: Rosetta 2 설치 또는 ARM 네이티브 사용

#### Linux 특수 문제
- **udev 규칙 미적용**: 시스템 재부팅 또는 재로그인
- **사용자 그룹 문제**: `newgrp dialout` 명령어 사용
- **커널 모듈 문제**: `sudo modprobe usbserial` 실행

## 📊 성능 최적화

### Windows 최적화
```cmd
# 전원 관리 설정
# 제어판 → 전원 옵션 → 고성능

# USB 전원 관리 비활성화
# 장치 관리자 → USB 컨트롤러 → 속성 → 전원 관리
```

### macOS 최적화
```bash
# USB 전원 관리 비활성화
sudo pmset -a usbpower 0

# 성능 모드 설정
sudo pmset -a highstandbythreshold 0
```

### Linux 최적화
```bash
# CPU 성능 모드 설정
sudo cpupower frequency-set -g performance

# USB 전원 관리 비활성화
echo 'ACTION=="add", SUBSYSTEM=="usb", ATTR{power/autosuspend}="-1"' | sudo tee /etc/udev/rules.d/99-usb-power.rules
```

## 📚 추가 자료

### 공식 문서
- [Python 공식 문서](https://docs.python.org/)
- [PySerial 문서](https://pyserial.readthedocs.io/)
- [Arduino 공식 사이트](https://www.arduino.cc/)

### 플랫폼별 문서
- [Windows 개발자 문서](https://docs.microsoft.com/en-us/windows/)
- [macOS 개발자 문서](https://developer.apple.com/macos/)
- [Linux 문서](https://www.kernel.org/doc/)

### 커뮤니티 지원
- [GitHub Issues](https://github.com/KNDG01001/SignGlove_HW/issues)
- [Stack Overflow](https://stackoverflow.com/)
- [Arduino 포럼](https://forum.arduino.cc/)

---

**🤟 SignGlove 크로스 플랫폼 시스템 - 모든 플랫폼에서 완벽한 수화 인식**
