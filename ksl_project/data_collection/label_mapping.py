# KSL 클래스 정의 및 라벨 매핑 시스템
import os
import re
from typing import Dict, List, Optional, Tuple

class KSLLabelMapper:
    def __init__(self):
        """34개 KSL 클래스 정의"""
        self.classes = {
            # 한글 자음 (14개)
            '한글_자음': ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 
                        'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'],
            
            # 한글 모음 (10개)  
            '한글_모음': ['ㅏ', 'ㅓ', 'ㅗ', 'ㅜ', 'ㅡ', 'ㅣ', 'ㅑ', 'ㅕ', 
                        'ㅛ', 'ㅠ'],
            
            # 숫자 (10개)
            '숫자': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        }
        
        # 전체 클래스 리스트 생성 (순서 보장)
        self.class_list = []
        for category in ['한글_자음', '한글_모음', '숫자']:
            self.class_list.extend(self.classes[category])
        
        # 클래스명 -> ID 매핑
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.class_list)}
        self.id_to_class = {idx: cls for cls, idx in self.class_to_id.items()}
        
        print(f"총 {len(self.class_list)}개 KSL 클래스 정의 완료")
        print(f"클래스 목록: {self.class_list}")
    
    def extract_label_from_filename(self, filename: str) -> Optional[int]:
        """
        파일명에서 라벨 추출
        예시: 'ㄱ_user1_session1.csv' -> 0 (ㄱ의 클래스 ID)
        """
        # 파일명에서 확장자 제거
        basename = os.path.splitext(filename)[0]
        
        # 다양한 패턴으로 클래스명 추출 시도
        patterns = [
            r'^([ㄱ-ㅎㅏ-ㅣ0-9])_',  # 클래스_user 형태
            r'^([ㄱ-ㅎㅏ-ㅣ0-9])',    # 클래스만 있는 경우
            r'_([ㄱ-ㅎㅏ-ㅣ0-9])_',   # _클래스_ 형태
            r'([ㄱ-ㅎㅏ-ㅣ0-9])'      # 아무 위치의 클래스
        ]
        
        for pattern in patterns:
            match = re.search(pattern, basename)
            if match:
                class_name = match.group(1)
                if class_name in self.class_to_id:
                    return self.class_to_id[class_name]
        
        # 패턴 매칭 실패 시 None 반환
        return None
    
    def create_metadata_file(self, csv_dir: str, output_path: str = None):
        """
        CSV 파일들을 스캔하여 메타데이터 파일 생성
        """
        if output_path is None:
            output_path = os.path.join(csv_dir, 'labels_metadata.txt')
        
        metadata = []
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        
        for filename in csv_files:
            label_id = self.extract_label_from_filename(filename)
            if label_id is not None:
                class_name = self.id_to_class[label_id]
                metadata.append(f"{filename}\t{label_id}\t{class_name}")
            else:
                metadata.append(f"{filename}\t-1\tunknown")
        
        # 메타데이터 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("filename\tlabel_id\tclass_name\n")
            for line in metadata:
                f.write(line + "\n")
        
        print(f"메타데이터 파일 생성: {output_path}")
        return output_path
    
    def get_class_distribution(self, csv_dir: str) -> Dict[str, int]:
        """클래스별 데이터 분포 확인"""
        distribution = {cls: 0 for cls in self.class_list}
        distribution['unknown'] = 0
        
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        
        for filename in csv_files:
            label_id = self.extract_label_from_filename(filename)
            if label_id is not None:
                class_name = self.id_to_class[label_id]
                distribution[class_name] += 1
            else:
                distribution['unknown'] += 1
        
        return distribution
    
    def create_sample_labeled_data(self, base_csv_path: str, output_dir: str, num_samples: int = 5):
        """
        기존 CSV 파일을 기반으로 샘플 라벨 데이터 생성
        (실제 프로젝트에서는 실제 수집된 데이터 사용)
        """
        import pandas as pd
        import numpy as np
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 기존 CSV 파일 로드
        base_df = pd.read_csv(base_csv_path, encoding='latin1')
        
        created_files = []
        
        # 각 클래스별로 샘플 파일 생성
        for i, class_name in enumerate(self.class_list[:num_samples]):  # 처음 5개 클래스만
            # 데이터에 약간의 노이즈 추가하여 변형
            modified_df = base_df.copy()
            
            # flex 센서에 클래스별 특성 반영 (예시)
            noise_factor = np.random.normal(1.0, 0.1, len(modified_df))
            for col in ['flex1', 'flex2', 'flex3', 'flex4', 'flex5']:
                if col in modified_df.columns:
                    modified_df[col] = modified_df[col] * noise_factor * (1 + i * 0.1)
            
            # 파일명에 클래스명 포함
            output_filename = f"{class_name}_sample_data.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            modified_df.to_csv(output_path, index=False, encoding='utf-8')
            created_files.append(output_filename)
            
            print(f"생성됨: {output_filename} (클래스: {class_name}, ID: {i})")
        
        return created_files

# 사용 예시
if __name__ == "__main__":
    mapper = KSLLabelMapper()
    
    # 클래스 정보 출력
    print("\n=== KSL 클래스 정보 ===")
    for category, classes in mapper.classes.items():
        print(f"{category}: {classes}")
    
    # 파일명에서 라벨 추출 테스트
    print("\n=== 라벨 추출 테스트 ===")
    test_files = ['ㄱ_user1.csv', 'ㅏ_session1.csv', '1_test.csv', 'unknown_file.csv']
    for filename in test_files:
        label_id = mapper.extract_label_from_filename(filename)
        if label_id is not None:
            class_name = mapper.id_to_class[label_id]
            print(f"{filename} -> ID: {label_id}, 클래스: {class_name}")
        else:
            print(f"{filename} -> 라벨 추출 실패")