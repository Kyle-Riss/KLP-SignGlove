import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class LetterResult:
    """개별 글자 인식 결과"""
    letter: str
    confidence: float
    timestamp: float

@dataclass
class WordResult:
    """단어 인식 결과"""
    word: str
    confidence: float
    letters: List[LetterResult]
    timestamp: float

class KoreanDictionary:
    """한국어 단어 사전"""
    
    def __init__(self):
        # 자주 사용되는 한국어 단어들
        self.common_words = [
            # 인사말
            "안녕하세요", "안녕히가세요", "안녕히계세요", "반갑습니다",
            "만나서반가워요", "오랜만입니다", "또만나요",
            
            # 감사/사과
            "감사합니다", "고맙습니다", "죄송합니다", "미안합니다",
            "사과드립니다", "양해해주세요",
            
            # 긍정/부정
            "네", "예", "맞습니다", "틀렸습니다", "아니요",
            "괜찮습니다", "좋습니다", "나쁩니다",
            
            # 질문
            "무엇입니까", "어떻게", "언제", "어디서", "왜",
            "몇시입니까", "얼마입니까",
            
            # 도움
            "도와주세요", "도움이필요합니다", "설명해주세요",
            "알려주세요", "가르쳐주세요",
            
            # 이해
            "이해했습니다", "모르겠습니다", "알겠습니다",
            "잘알겠습니다", "이해가안됩니다",
            
            # 기타
            "맛있습니다", "맛없습니다", "힘듭니다", "쉽습니다",
            "빠릅니다", "느립니다", "크습니다", "작습니다"
        ]
        
        # 단어 길이별로 정리
        self.words_by_length = {}
        for word in self.common_words:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = []
            self.words_by_length[length].append(word)
    
    def find_complete_word(self, partial_word: str) -> Optional[str]:
        """부분 단어로 완성된 단어 찾기"""
        if not partial_word:
            return None
            
        # 정확히 일치하는 단어 찾기
        if partial_word in self.common_words:
            return partial_word
        
        # 부분 단어로 시작하는 완성된 단어 찾기
        candidates = []
        for word in self.common_words:
            if word.startswith(partial_word):
                candidates.append(word)
        
        if candidates:
            # 가장 짧은 완성된 단어 반환
            return min(candidates, key=len)
        
        return None
    
    def get_suggestions(self, partial_word: str, max_suggestions: int = 3) -> List[str]:
        """단어 제안"""
        suggestions = []
        for word in self.common_words:
            if word.startswith(partial_word) and len(suggestions) < max_suggestions:
                suggestions.append(word)
        return suggestions

class WordRecognitionSystem:
    """실시간 단어 인식 시스템"""
    
    def __init__(self):
        self.dictionary = KoreanDictionary()
        
        # 버퍼 설정
        self.letter_buffer = deque(maxlen=20)  # 최근 20개 글자
        self.word_history = []
        
        # 설정
        self.confidence_threshold = 0.8
        self.word_timeout = 3.0  # 3초 이상 간격이면 새 단어
        self.min_word_length = 2
        self.max_word_length = 10
        
        # 상태
        self.current_word = ""
        self.last_letter_time = 0
        self.is_collecting = False
    
    def add_letter(self, letter: str, confidence: float, timestamp: float = None) -> Optional[WordResult]:
        """글자를 직접 추가하고 단어 인식 수행"""
        if timestamp is None:
            timestamp = time.time()
            
        # 글자 결과 생성
        letter_result = LetterResult(
            letter=letter,
            confidence=confidence,
            timestamp=timestamp
        )
        
        # 단어 인식 처리
        return self.process_letter(letter_result)
    
    def process_letter(self, letter_result: LetterResult) -> Optional[WordResult]:
        """글자를 처리하고 단어 완성 체크"""
        current_time = letter_result.timestamp
        
        # 신뢰도가 낮으면 무시
        if letter_result.confidence < self.confidence_threshold:
            return None
        
        # 시간 간격이 길면 새 단어 시작
        if current_time - self.last_letter_time > self.word_timeout:
            if self.current_word:
                # 이전 단어 완성
                completed_word = self.complete_current_word()
                if completed_word:
                    self.word_history.append(completed_word)
                    self.current_word = ""
                    self.is_collecting = False
        
        # 글자 추가
        if not self.is_collecting:
            self.is_collecting = True
        
        self.current_word += letter_result.letter
        self.letter_buffer.append(letter_result)
        self.last_letter_time = current_time
        
        # 단어 길이 제한
        if len(self.current_word) > self.max_word_length:
            self.current_word = self.current_word[:self.max_word_length]
        
        # 단어 완성 체크
        if len(self.current_word) >= self.min_word_length:
            completed_word = self.dictionary.find_complete_word(self.current_word)
            if completed_word:
                # 완성된 단어 생성
                word_result = WordResult(
                    word=completed_word,
                    confidence=sum(l.confidence for l in self.letter_buffer) / len(self.letter_buffer),
                    letters=list(self.letter_buffer),
                    timestamp=current_time
                )
                
                # 상태 초기화
                self.word_history.append(completed_word)
                self.current_word = ""
                self.letter_buffer.clear()
                self.is_collecting = False
                
                return word_result
        
        return None
    
    def complete_current_word(self) -> Optional[str]:
        """현재 단어 완성"""
        if not self.current_word:
            return None
        
        # 사전에서 완성된 단어 찾기
        completed_word = self.dictionary.find_complete_word(self.current_word)
        if completed_word:
            return completed_word
        
        # 완성되지 않았지만 의미있는 부분 단어 반환
        if len(self.current_word) >= self.min_word_length:
            return self.current_word
        
        return None
    
    def clear_current_word(self):
        """현재 단어 초기화"""
        self.current_word = ""
        self.letter_buffer.clear()
        self.is_collecting = False
        self.last_letter_time = 0
    
    def get_current_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'current_word': self.current_word,
            'is_collecting': self.is_collecting,
            'word_history': self.word_history[-5:],
            'suggestions': self.dictionary.get_suggestions(self.current_word),
            'buffer_size': len(self.letter_buffer),
            'time_since_last_letter': time.time() - self.last_letter_time if self.last_letter_time > 0 else 0
        }

# 사용 예시
if __name__ == "__main__":
    # 단어 인식 시스템 초기화
    word_system = WordRecognitionSystem()
    
    print("🎯 단어 인식 시스템 테스트")
    print("=" * 50)
    
    # "안녕하세요" 테스트
    test_letters = [
        ("ㅇ", 0.95),
        ("ㅏ", 0.92),
        ("ㄴ", 0.88),
        ("ㄴ", 0.90),
        ("ㅕ", 0.87),
        ("ㅇ", 0.93),
        ("ㅎ", 0.91),
        ("ㅏ", 0.89),
        ("ㅅ", 0.94),
        ("ㅔ", 0.86),
        ("ㅇ", 0.92),
        ("ㅛ", 0.88)
    ]
    
    for i, (letter, confidence) in enumerate(test_letters):
        print(f"\n📝 글자 {i+1}: {letter} (신뢰도: {confidence:.2f})")
        
        result = word_system.add_letter(letter, confidence)
        
        if result:
            print(f"✅ 단어 완성: {result.word}")
            print(f"   신뢰도: {result.confidence:.2f}")
            print(f"   글자 수: {len(result.letters)}")
        else:
            status = word_system.get_current_status()
            print(f"📊 현재 단어: {status['current_word']}")
            print(f"💡 제안: {status['suggestions']}")
            print(f"🔍 사전 검색: {word_system.dictionary.find_complete_word(status['current_word'])}")
    
    # 최종 상태 출력
    final_status = word_system.get_current_status()
    print(f"\n🎯 최종 결과:")
    print(f"   완성된 단어들: {final_status['word_history']}")
    print(f"   현재 단어: {final_status['current_word']}")
    print(f"   사전에서 찾기: {word_system.dictionary.find_complete_word(final_status['current_word'])}")
