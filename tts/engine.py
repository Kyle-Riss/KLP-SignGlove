try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

import threading
import queue
import time
from typing import Dict, List, Optional, Callable
import platform

class KoreanTTSEngine:
    """
    한국어 Text-to-Speech 엔진
    - 실시간 수어 인식 결과를 음성으로 출력
    - 다양한 TTS 엔진 지원 (pyttsx3, macOS say 등)
    - 비동기 음성 출력 지원
    """
    
    def __init__(self, engine_type: str = 'auto', rate: int = 200, volume: float = 0.8):
        """
        Args:
            engine_type: TTS 엔진 타입 ('auto', 'pyttsx3', 'macos', 'espeak')
            rate: 말하기 속도 (words per minute)
            volume: 음량 (0.0 ~ 1.0)
        """
        self.engine_type = engine_type
        self.rate = rate
        self.volume = volume
        
        # 음성 출력 큐
        self.speech_queue = queue.Queue()
        self.is_running = False
        
        # TTS 엔진 초기화
        self.engine = self._initialize_engine()
        
        # 한국어 발음 매핑 (자음)
        self.korean_pronunciations = {
            'ㄱ': '기역',
            'ㄴ': '니은',
            'ㄷ': '디귿',
            'ㄹ': '리을',
            'ㅁ': '미음',
            'ㅂ': '비읍',
            'ㅅ': '시옷',
            'ㅇ': '이응',
            'ㅈ': '지읒',
            'ㅊ': '치읓',
            'ㅋ': '키읔',
            'ㅌ': '티읕',
            'ㅍ': '피읖',
            'ㅎ': '히읗',
            # 모음
            'ㅏ': '아',
            'ㅓ': '어',
            'ㅗ': '오',
            'ㅜ': '우',
            'ㅡ': '으',
            'ㅣ': '이',
            'ㅑ': '야',
            'ㅕ': '여',
            'ㅛ': '요',
            'ㅠ': '유',
            # 숫자
            '1': '일', '2': '이', '3': '삼', '4': '사', '5': '오',
            '6': '육', '7': '칠', '8': '팔', '9': '구', '0': '영'
        }
        
        print(f"TTS 엔진 초기화 완료: {self.engine_type}")
    
    def _initialize_engine(self):
        """TTS 엔진 초기화"""
        system = platform.system().lower()
        
        if self.engine_type == 'auto':
            if system == 'darwin':  # macOS
                return self._init_macos_engine()
            else:
                return self._init_pyttsx3_engine()
        elif self.engine_type == 'macos' and system == 'darwin':
            return self._init_macos_engine()
        elif self.engine_type == 'pyttsx3':
            return self._init_pyttsx3_engine()
        else:
            # 기본값으로 pyttsx3 사용
            return self._init_pyttsx3_engine()
    
    def _init_pyttsx3_engine(self):
        """pyttsx3 엔진 초기화"""
        try:
            if not PYTTSX3_AVAILABLE:
                print("pyttsx3 라이브러리가 설치되어 있지 않습니다. 콘솔 출력으로 대체합니다.")
                self.engine_type = 'console'
                return None
                
            import pyttsx3
            engine = pyttsx3.init()
            
            # 설정 적용
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            
            # 한국어 음성 찾기
            voices = engine.getProperty('voices')
            korean_voice = None
            
            for voice in voices:
                if 'korean' in voice.name.lower() or 'ko' in voice.id.lower():
                    korean_voice = voice.id
                    break
            
            if korean_voice:
                engine.setProperty('voice', korean_voice)
                print(f"한국어 음성 설정: {korean_voice}")
            else:
                print("한국어 음성을 찾을 수 없어 기본 음성을 사용합니다.")
            
            self.engine_type = 'pyttsx3'
            return engine
            
        except Exception as e:
            print(f"pyttsx3 초기화 오류: {e}")
            return None
    
    def _init_macos_engine(self):
        """macOS say 명령 엔진 초기화"""
        import subprocess
        
        try:
            # 사용 가능한 음성 확인
            result = subprocess.run(['say', '-v', '?'], 
                                  capture_output=True, text=True)
            
            korean_voices = []
            for line in result.stdout.split('\n'):
                if 'ko_' in line or 'korean' in line.lower():
                    voice_name = line.split()[0]
                    korean_voices.append(voice_name)
            
            if korean_voices:
                self.macos_voice = korean_voices[0]
                print(f"macOS 한국어 음성 설정: {self.macos_voice}")
            else:
                self.macos_voice = None
                print("macOS 한국어 음성을 찾을 수 없어 기본 음성을 사용합니다.")
            
            self.engine_type = 'macos'
            return 'macos_say'
            
        except Exception as e:
            print(f"macOS TTS 초기화 오류: {e}")
            return None
    
    def speak(self, text: str, blocking: bool = False):
        """
        텍스트 음성 출력
        Args:
            text: 출력할 텍스트
            blocking: True면 음성 출력이 끝날 때까지 대기
        """
        if not text:
            return
        
        # 한국어 문자 발음으로 변환
        pronunciation = self._convert_to_pronunciation(text)
        
        if blocking:
            self._speak_immediately(pronunciation)
        else:
            # 큐에 추가하여 비동기 출력
            self.speech_queue.put(pronunciation)
    
    def _convert_to_pronunciation(self, text: str) -> str:
        """한국어 문자를 발음 가능한 텍스트로 변환"""
        result = []
        
        for char in text:
            if char in self.korean_pronunciations:
                result.append(self.korean_pronunciations[char])
            else:
                result.append(char)
        
        return ' '.join(result)
    
    def _speak_immediately(self, text: str):
        """즉시 음성 출력"""
        try:
            if self.engine_type == 'pyttsx3' and self.engine and PYTTSX3_AVAILABLE:
                self.engine.say(text)
                self.engine.runAndWait()
            
            elif self.engine_type == 'macos':
                import subprocess
                cmd = ['say']
                
                if hasattr(self, 'macos_voice') and self.macos_voice:
                    cmd.extend(['-v', self.macos_voice])
                
                # 말하기 속도 설정
                cmd.extend(['-r', str(self.rate)])
                cmd.append(text)
                
                subprocess.run(cmd, check=True)
            
            else:
                print(f"🔊 TTS: {text}")  # 폴백: 콘솔 출력
                
        except Exception as e:
            print(f"음성 출력 오류: {e}")
            print(f"🔊 TTS: {text}")  # 폴백: 콘솔 출력
    
    def start_async_speaking(self):
        """비동기 음성 출력 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        
        def speech_loop():
            while self.is_running:
                try:
                    # 큐에서 텍스트 가져오기
                    text = self.speech_queue.get(timeout=0.5)
                    self._speak_immediately(text)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"음성 출력 루프 오류: {e}")
                    continue
        
        self.speech_thread = threading.Thread(target=speech_loop, daemon=True)
        self.speech_thread.start()
        print("비동기 음성 출력 시작됨")
    
    def stop_async_speaking(self):
        """비동기 음성 출력 중지"""
        self.is_running = False
        if hasattr(self, 'speech_thread'):
            self.speech_thread.join(timeout=1.0)
        print("비동기 음성 출력 중지됨")
    
    def speak_prediction_result(self, prediction_result: Dict):
        """
        예측 결과를 음성으로 출력
        Args:
            prediction_result: 예측 결과 딕셔너리
        """
        if 'predicted_class' not in prediction_result:
            return
        
        predicted_class = prediction_result['predicted_class']
        confidence = prediction_result.get('confidence', 0.0)
        
        # 신뢰도가 높은 경우에만 출력
        if confidence >= 0.8:
            self.speak(predicted_class)
        elif confidence >= 0.6:
            # 신뢰도가 중간인 경우 불확실함을 표현
            self.speak(f"{predicted_class} 같아요")
    
    def speak_with_confidence(self, text: str, confidence: float):
        """신뢰도에 따른 음성 출력"""
        if confidence >= 0.9:
            self.speak(text)
        elif confidence >= 0.7:
            self.speak(f"{text} 같습니다")
        elif confidence >= 0.5:
            self.speak(f"{text} 인 것 같아요")
        else:
            self.speak("잘 모르겠어요")
    
    def test_pronunciation(self):
        """발음 테스트"""
        test_chars = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
        
        print("TTS 발음 테스트 시작...")
        for char in test_chars:
            print(f"발음 테스트: {char}")
            self.speak(char, blocking=True)
            time.sleep(0.5)
        
        print("TTS 발음 테스트 완료")
    
    def set_rate(self, rate: int):
        """말하기 속도 설정"""
        self.rate = max(50, min(400, rate))  # 50-400 범위로 제한
        
        if self.engine_type == 'pyttsx3' and self.engine:
            self.engine.setProperty('rate', self.rate)
        
        print(f"말하기 속도 설정: {self.rate}")
    
    def set_volume(self, volume: float):
        """음량 설정"""
        self.volume = max(0.0, min(1.0, volume))
        
        if self.engine_type == 'pyttsx3' and self.engine:
            self.engine.setProperty('volume', self.volume)
        
        print(f"음량 설정: {self.volume}")
    
    def clear_queue(self):
        """음성 출력 큐 비우기"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
        print("음성 출력 큐 초기화 완료")
    
    def get_queue_size(self) -> int:
        """대기 중인 음성 출력 개수 반환"""
        return self.speech_queue.qsize()


# 간단한 사용 예시
if __name__ == "__main__":
    # TTS 엔진 테스트
    tts = KoreanTTSEngine()
    
    # 비동기 모드 시작
    tts.start_async_speaking()
    
    # 테스트 발음
    test_predictions = [
        {'predicted_class': 'ㄱ', 'confidence': 0.95},
        {'predicted_class': 'ㄴ', 'confidence': 0.88},
        {'predicted_class': 'ㄷ', 'confidence': 0.72},
        {'predicted_class': 'ㄹ', 'confidence': 0.65},
        {'predicted_class': 'ㅁ', 'confidence': 0.45},
    ]
    
    for pred in test_predictions:
        print(f"예측: {pred['predicted_class']} (신뢰도: {pred['confidence']:.2f})")
        tts.speak_prediction_result(pred)
        time.sleep(2)
    
    # 비동기 모드 중지
    time.sleep(5)
    tts.stop_async_speaking()
