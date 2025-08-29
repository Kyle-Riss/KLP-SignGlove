#!/usr/bin/env python3
"""
크로스 플랫폼 유틸리티
Windows, macOS, Linux 호환성을 위한 도구들
"""

import sys
import os
import glob
import logging
from typing import List, Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

class PlatformUtils:
    """크로스 플랫폼 유틸리티 클래스"""
    
    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """현재 플랫폼 정보 반환"""
        return {
            'platform': sys.platform,
            'os_name': os.name,
            'python_version': sys.version,
            'architecture': sys.maxsize > 2**32 and '64bit' or '32bit'
        }
    
    @staticmethod
    def is_windows() -> bool:
        """Windows 플랫폼인지 확인"""
        return sys.platform.startswith('win')
    
    @staticmethod
    def is_macos() -> bool:
        """macOS 플랫폼인지 확인"""
        return sys.platform.startswith('darwin')
    
    @staticmethod
    def is_linux() -> bool:
        """Linux 플랫폼인지 확인"""
        return sys.platform.startswith('linux')
    
    @staticmethod
    def get_serial_ports() -> List[str]:
        """사용 가능한 시리얼 포트 목록 반환"""
        ports = []
        
        if PlatformUtils.is_windows():
            ports = PlatformUtils._get_windows_ports()
        elif PlatformUtils.is_macos():
            ports = PlatformUtils._get_macos_ports()
        elif PlatformUtils.is_linux():
            ports = PlatformUtils._get_linux_ports()
        else:
            logger.warning(f"지원되지 않는 플랫폼: {sys.platform}")
        
        return ports
    
    @staticmethod
    def _get_windows_ports() -> List[str]:
        """Windows 시리얼 포트 목록"""
        ports = []
        
        # 기본 COM 포트들
        for i in range(1, 21):
            ports.append(f'COM{i}')
        
        # 레지스트리에서 실제 사용 가능한 포트들
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                    if "COM" in value and value not in ports:
                        ports.append(value)
                    i += 1
                except WindowsError:
                    break
            winreg.CloseKey(key)
        except Exception as e:
            logger.debug(f"Windows 레지스트리 검색 실패: {e}")
        
        return ports
    
    @staticmethod
    def _get_macos_ports() -> List[str]:
        """macOS 시리얼 포트 목록"""
        ports = []
        
        # macOS에서 사용되는 시리얼 포트 패턴들
        patterns = [
            '/dev/tty.usb*',
            '/dev/tty.usbserial*',
            '/dev/tty.usbmodem*',
            '/dev/tty.usbmodem*',
            '/dev/ttyACM*',
            '/dev/tty.SLAB_USBtoUART*'
        ]
        
        for pattern in patterns:
            ports.extend(glob.glob(pattern))
        
        return ports
    
    @staticmethod
    def _get_linux_ports() -> List[str]:
        """Linux 시리얼 포트 목록"""
        ports = []
        
        # Linux에서 사용되는 시리얼 포트 패턴들
        patterns = [
            '/dev/ttyUSB*',
            '/dev/ttyACM*',
            '/dev/tty.usb*',
            '/dev/ttyS*'
        ]
        
        for pattern in patterns:
            ports.extend(glob.glob(pattern))
        
        return ports
    
    @staticmethod
    def get_serial_config() -> Dict[str, Any]:
        """플랫폼별 시리얼 설정 반환"""
        if PlatformUtils.is_windows():
            return {
                'timeout': 1,
                'write_timeout': 1,
                'bytesize': 8,
                'parity': 'N',
                'stopbits': 1
            }
        else:
            return {
                'timeout': 1,
                'bytesize': 8,
                'parity': 'N',
                'stopbits': 1
            }
    
    @staticmethod
    def test_port_connection(port: str, baudrate: int = 115200) -> bool:
        """포트 연결 테스트"""
        try:
            import serial
            conn = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1,
                **PlatformUtils.get_serial_config()
            )
            conn.close()
            return True
        except Exception as e:
            logger.debug(f"포트 {port} 연결 테스트 실패: {e}")
            return False
    
    @staticmethod
    def find_arduino_port() -> Optional[str]:
        """아두이노 포트 자동 감지"""
        ports = PlatformUtils.get_serial_ports()
        logger.info(f"검색된 포트들: {ports}")
        
        for port in ports:
            if PlatformUtils.test_port_connection(port):
                logger.info(f"아두이노 포트 발견: {port}")
                return port
        
        logger.warning("아두이노 포트를 찾을 수 없습니다.")
        return None
    
    @staticmethod
    def get_file_path_separator() -> str:
        """파일 경로 구분자 반환"""
        return os.path.sep
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """경로 정규화"""
        return os.path.normpath(path)
    
    @staticmethod
    def get_home_directory() -> str:
        """홈 디렉토리 경로 반환"""
        return os.path.expanduser("~")
    
    @staticmethod
    def create_directory_if_not_exists(path: str) -> bool:
        """디렉토리가 없으면 생성"""
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"디렉토리 생성 실패: {e}")
            return False
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """시스템 정보 반환"""
        import platform
        
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation()
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """메모리 정보 반환"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            }
        except ImportError:
            return {'error': 'psutil not available'}
    
    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """CPU 정보 반환"""
        try:
            import psutil
            return {
                'count': psutil.cpu_count(),
                'percent': psutil.cpu_percent(interval=1),
                'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
        except ImportError:
            return {'error': 'psutil not available'}

def main():
    """테스트 함수"""
    print("🖥️ 크로스 플랫폼 유틸리티 테스트")
    print("="*50)
    
    # 플랫폼 정보
    platform_info = PlatformUtils.get_platform_info()
    print(f"플랫폼: {platform_info['platform']}")
    print(f"OS: {platform_info['os_name']}")
    print(f"Python: {platform_info['python_version']}")
    print(f"아키텍처: {platform_info['architecture']}")
    
    # 시리얼 포트
    ports = PlatformUtils.get_serial_ports()
    print(f"\n시리얼 포트: {ports}")
    
    # 아두이노 포트
    arduino_port = PlatformUtils.find_arduino_port()
    if arduino_port:
        print(f"아두이노 포트: {arduino_port}")
    else:
        print("아두이노 포트: 없음")
    
    # 시스템 정보
    system_info = PlatformUtils.get_system_info()
    print(f"\n시스템: {system_info['system']} {system_info['release']}")
    print(f"프로세서: {system_info['processor']}")

if __name__ == "__main__":
    main()
