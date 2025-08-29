"""
KSL 라벨 매핑 클래스
파일명에서 자동으로 라벨을 추출하고 관리
24개 클래스 지원 (14개 자음 + 10개 모음)
"""

import re
import os

class KSLLabelMapper:
    def __init__(self):
        """한국어 자음/모음 기반 라벨 매퍼 초기화"""
        # 24개 클래스 매핑 (14개 자음 + 10개 모음)
        self.class_to_id = {
            # 자음 (14개)
            'ㄱ': 0, 'ㄴ': 1, 'ㄷ': 2, 'ㄹ': 3, 'ㅁ': 4,
            'ㅂ': 5, 'ㅅ': 6, 'ㅇ': 7, 'ㅈ': 8, 'ㅊ': 9,
            'ㅋ': 10, 'ㅌ': 11, 'ㅍ': 12, 'ㅎ': 13,
            # 모음 (10개)
            'ㅏ': 14, 'ㅑ': 15, 'ㅓ': 16, 'ㅕ': 17, 'ㅗ': 18,
            'ㅛ': 19, 'ㅜ': 20, 'ㅠ': 21, 'ㅡ': 22, 'ㅣ': 23
        }
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        
        print(f"🎯 24개 클래스 라벨 매퍼 초기화 완료")
        print(f"  📝 자음: {list(self.class_to_id.keys())[:14]}")
        print(f"  📝 모음: {list(self.class_to_id.keys())[14:]}")
        
    def extract_label_from_filename(self, filename: str) -> int:
        """
        파일명에서 라벨 추출
        
        Args:
            filename: 분석할 파일명
            
        Returns:
            라벨 ID (없으면 None)
        """
        try:
            # 파일명에서 확장자 제거
            base_name = os.path.splitext(filename)[0]
            
            # 한국어 자음/모음 패턴 찾기
            for korean_char, label_id in self.class_to_id.items():
                if korean_char in base_name:
                    return label_id
            
            # 패턴이 없으면 None 반환
            return None
            
        except Exception as e:
            print(f"라벨 추출 오류: {e}")
            return None
    
    def get_class_name(self, label_id: int) -> str:
        """라벨 ID로 클래스명 반환"""
        return self.id_to_class.get(label_id, f"unknown_{label_id}")
    
    def get_label_id(self, class_name: str) -> int:
        """클래스명으로 라벨 ID 반환"""
        return self.class_to_id.get(class_name, -1)
    
    def get_all_classes(self) -> list:
        """모든 클래스 반환"""
        return list(self.class_to_id.keys())
    
    def get_num_classes(self) -> int:
        """총 클래스 수 반환"""
        return len(self.class_to_id)
    
    def get_consonants(self) -> list:
        """자음만 반환"""
        return ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    
    def get_vowels(self) -> list:
        """모음만 반환"""
        return ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    def is_consonant(self, class_name: str) -> bool:
        """자음인지 확인"""
        return class_name in self.get_consonants()
    
    def is_vowel(self, class_name: str) -> bool:
        """모음인지 확인"""
        return class_name in self.get_vowels()

# 테스트용
if __name__ == "__main__":
    mapper = KSLLabelMapper()
    
    test_files = [
        "ㄱ_unified_data_021.csv",
        "ㄴ_unified_data_026.csv", 
        "ㄷ_unified_data_024.csv",
        "ㄹ_unified_data_002.csv",
        "ㅁ_unified_data_028.csv",
        "ㅂ_unified_data_030.csv",
        "ㅅ_unified_data_027.csv",
        "ㅇ_unified_data_025.csv",
        "ㅈ_unified_data_004.csv",
        "ㅊ_unified_data_011.csv",
        "ㅋ_unified_data_023.csv",
        "ㅌ_unified_data_001.csv",
        "ㅍ_unified_data_022.csv",
        "ㅎ_unified_data_007.csv",
        "ㅏ_unified_data_012.csv",
        "ㅑ_unified_data_003.csv",
        "ㅓ_unified_data_013.csv",
        "ㅕ_unified_data_006.csv",
        "ㅗ_unified_data_018.csv",
        "ㅛ_unified_data_000.csv",
        "ㅜ_unified_data_014.csv",
        "ㅠ_unified_data_016.csv",
        "ㅡ_unified_data_017.csv",
        "ㅣ_unified_data_005.csv",
        "unknown_file.csv"
    ]
    
    print("=== 24개 클래스 라벨 매핑 테스트 ===")
    success_count = 0
    for filename in test_files:
        label_id = mapper.extract_label_from_filename(filename)
        if label_id is not None:
            class_name = mapper.get_class_name(label_id)
            char_type = "자음" if mapper.is_consonant(class_name) else "모음"
            print(f"{filename} -> 라벨 {label_id} (클래스: {class_name}, 타입: {char_type})")
            success_count += 1
        else:
            print(f"{filename} -> 라벨 추출 실패")
    
    print(f"\n✅ 성공: {success_count}/{len(test_files)-1}개 파일")
    print(f"📊 총 클래스 수: {mapper.get_num_classes()}")
    print(f"🔤 자음: {len(mapper.get_consonants())}개")
    print(f"🅰️ 모음: {len(mapper.get_vowels())}개")

