@echo off
REM SignGlove Windows 설정 스크립트
REM Windows 사용자를 위한 자동 설정 도구

echo ========================================
echo    SignGlove Windows 설정 도구
echo ========================================
echo.

REM Python 설치 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되지 않았습니다.
    echo Python 3.8+를 설치해주세요: https://www.python.org/downloads/
    echo 설치 시 "Add Python to PATH" 옵션을 체크해주세요.
    pause
    exit /b 1
)

echo [INFO] Python 설치 확인됨
python --version

REM pip 업그레이드
echo.
echo [INFO] pip 업그레이드 중...
python -m pip install --upgrade pip

REM 필수 패키지 설치
echo.
echo [INFO] 필수 패키지 설치 중...
pip install pyserial requests numpy pandas

REM 선택적 패키지 설치
echo.
echo [INFO] 선택적 패키지 설치 중...
pip install psutil

REM Arduino 드라이버 확인
echo.
echo [INFO] Arduino 드라이버 확인 중...
echo 다음 단계를 따라 Arduino 드라이버를 설치하세요:
echo 1. https://www.arduino.cc/en/software 에서 Arduino IDE 다운로드
echo 2. Arduino IDE 설치 (드라이버 포함)
echo 3. SignGlove_HW를 USB로 연결
echo 4. 장치 관리자에서 포트 확인

REM 포트 확인 도구
echo.
echo [INFO] 사용 가능한 COM 포트 확인 중...
powershell -Command "Get-WmiObject -Class Win32_SerialPort | Select-Object Name, DeviceID, Description"

echo.
echo ========================================
echo    설정 완료!
echo ========================================
echo.
echo 다음 명령어로 시스템을 시작하세요:
echo 1. API 서버: python server/main.py
echo 2. 통합 클라이언트: python integration/signglove_client.py
echo.
echo 문제가 있으면 다음을 확인하세요:
echo - Arduino IDE가 설치되어 있는지
echo - SignGlove_HW가 연결되어 있는지
echo - COM 포트가 정상적으로 인식되는지
echo.
pause
