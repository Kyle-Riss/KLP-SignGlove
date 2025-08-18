"""
KSL 라벨 매핑 클래스
파일명에서 자동으로 라벨을 추출하고 관리
"""

import re
import os

class KSLLabelMapper:
    def __init__(self):
        """한국어 자음 기반 라벨 매퍼 초기화"""
        # 기본 자음 매핑
        self.class_to_id = {
            'ㄱ': 0, 'ㄴ': 1, 'ㄷ': 2, 'ㄹ': 3, 'ㅁ': 4
        }
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        
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
            
            # 한국어 자음 패턴 찾기
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

# 테스트용
if __name__ == "__main__":
    mapper = KSLLabelMapper()
    
    test_files = [
        "ㄱ_sample_data.csv",
        "ㄴ_sample_data.csv", 
        "ㄷ_sample_data.csv",
        "ㄹ_sample_data.csv",
        "ㅁ_sample_data.csv",
        "test_madgwick_sample_data.csv",
        "unknown_file.csv"
    ]
    
    print("=== 라벨 매핑 테스트 ===")
    for filename in test_files:
        label_id = mapper.extract_label_from_filename(filename)
        if label_id is not None:
            class_name = mapper.get_class_name(label_id)
            print(f"{filename} -> 라벨 {label_id} (클래스: {class_name})")
        else:
            print(f"{filename} -> 라벨 추출 실패")
