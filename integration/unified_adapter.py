"""
SignGlove_HW Unified 데이터셋 어댑터
Unified 데이터셋을 KLP-SignGlove 프로젝트 형식으로 변환
"""

import pandas as pd
import numpy as np
import os
import re
from typing import Dict, List, Tuple
import glob

class UnifiedDataAdapter:
    """SignGlove_HW Unified 데이터셋 어댑터"""
    
    def __init__(self):
        self.unified_data_dir = "unified_dataset"
        self.output_dir = "integrations/SignGlove_HW"
        
    def download_unified_dataset(self, max_files_per_class: int = 5):
        """Unified 데이터셋 다운로드"""
        print("📥 SignGlove_HW Unified 데이터셋 다운로드 중...")
        
        # GitHub API로 파일 목록 가져오기
        import requests
        import json
        
        api_url = "https://api.github.com/repos/KNDG01001/SignGlove_HW/contents/unified"
        response = requests.get(api_url)
        
        if response.status_code != 200:
            print(f"❌ API 요청 실패: {response.status_code}")
            return False
            
        files = response.json()
        csv_files = [f for f in files if f['name'].endswith('.csv')]
        
        # 클래스별로 파일 그룹화
        class_files = {}
        for file_info in csv_files:
            filename = file_info['name']
            # 파일명에서 클래스 추출: episode_20250819_152212_ㄱ.csv -> ㄱ
            match = re.search(r'_([ㄱ-ㅎㅏ-ㅣ])\.csv$', filename)
            if match:
                class_name = match.group(1)
                if class_name not in class_files:
                    class_files[class_name] = []
                class_files[class_name].append(file_info)
        
        print(f"📊 발견된 클래스: {list(class_files.keys())}")
        
        # 각 클래스별로 파일 다운로드
        downloaded_count = 0
        for class_name, file_list in class_files.items():
            print(f"  📁 {class_name} 클래스: {len(file_list)}개 파일")
            
            # 각 클래스당 최대 파일 수만큼 다운로드
            for i, file_info in enumerate(file_list[:max_files_per_class]):
                filename = file_info['name']
                download_url = file_info['download_url']
                
                output_path = os.path.join(self.unified_data_dir, filename)
                
                print(f"    📥 다운로드: {filename}")
                response = requests.get(download_url)
                
                if response.status_code == 200:
                    os.makedirs(self.unified_data_dir, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    downloaded_count += 1
                else:
                    print(f"    ❌ 다운로드 실패: {filename}")
        
        print(f"✅ 총 {downloaded_count}개 파일 다운로드 완료")
        return True
    
    def convert_unified_to_ksl_format(self, input_file: str, output_file: str) -> pd.DataFrame:
        """Unified 형식을 KSL 형식으로 변환"""
        try:
            # Unified 데이터 로드
            df = pd.read_csv(input_file)
            print(f"  📄 로드: {os.path.basename(input_file)} ({len(df)}개 샘플)")
            
            # 컬럼 매핑
            column_mapping = {
                'timestamp_ms': 'timestamp(ms)',
                'pitch': 'pitch(°)',
                'roll': 'roll(°)', 
                'yaw': 'yaw(°)',
                'flex1': 'flex1',
                'flex2': 'flex2',
                'flex3': 'flex3',
                'flex4': 'flex4',
                'flex5': 'flex5'
            }
            
            # 필요한 컬럼만 선택하고 이름 변경
            ksl_df = df[list(column_mapping.keys())].copy()
            ksl_df.columns = list(column_mapping.values())
            
            # 타임스탬프를 밀리초로 변환 (필요시)
            if 'timestamp(ms)' in ksl_df.columns:
                ksl_df['timestamp(ms)'] = ksl_df['timestamp(ms)'].astype(float)
            
            # KSL 형식으로 저장
            ksl_df.to_csv(output_file, index=False)
            print(f"  ✅ 변환 완료: {os.path.basename(output_file)}")
            
            return ksl_df
            
        except Exception as e:
            print(f"  ❌ 변환 실패: {e}")
            return pd.DataFrame()
    
    def process_all_unified_data(self):
        """모든 Unified 데이터를 KSL 형식으로 변환"""
        print("🔄 Unified 데이터를 KSL 형식으로 변환 중...")
        
        # Unified 데이터 파일들 찾기
        unified_files = glob.glob(os.path.join(self.unified_data_dir, "*.csv"))
        
        if not unified_files:
            print("❌ Unified 데이터 파일이 없습니다.")
            return False
        
        converted_count = 0
        class_stats = {}
        
        for unified_file in unified_files:
            filename = os.path.basename(unified_file)
            
            # 파일명에서 클래스 추출
            match = re.search(r'_([ㄱ-ㅎㅏ-ㅣ])\.csv$', filename)
            if not match:
                continue
                
            class_name = match.group(1)
            
            # 출력 파일명 생성
            output_filename = f"{class_name}_unified_data_{converted_count:03d}.csv"
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 변환 실행
            converted_df = self.convert_unified_to_ksl_format(unified_file, output_path)
            
            if not converted_df.empty:
                converted_count += 1
                if class_name not in class_stats:
                    class_stats[class_name] = 0
                class_stats[class_name] += 1
        
        print(f"✅ 총 {converted_count}개 파일 변환 완료")
        print("📊 클래스별 변환 통계:")
        for class_name, count in class_stats.items():
            print(f"  {class_name}: {count}개")
        
        return True
    
    def analyze_unified_data(self):
        """Unified 데이터 분석"""
        print("📊 Unified 데이터 분석 중...")
        
        unified_files = glob.glob(os.path.join(self.unified_data_dir, "*.csv"))
        
        if not unified_files:
            print("❌ 분석할 데이터가 없습니다.")
            return
        
        class_data = {}
        
        for file_path in unified_files[:5]:  # 처음 5개 파일만 분석
            filename = os.path.basename(file_path)
            match = re.search(r'_([ㄱ-ㅎㅏ-ㅣ])\.csv$', filename)
            
            if match:
                class_name = match.group(1)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    if class_name not in class_data:
                        class_data[class_name] = {
                            'samples': [],
                            'columns': list(df.columns)
                        }
                    
                    class_data[class_name]['samples'].append({
                        'file': filename,
                        'rows': len(df),
                        'pitch_range': (df['pitch'].min(), df['pitch'].max()),
                        'roll_range': (df['roll'].min(), df['roll'].max()),
                        'yaw_range': (df['yaw'].min(), df['yaw'].max()),
                        'flex_ranges': [
                            (df['flex1'].min(), df['flex1'].max()),
                            (df['flex2'].min(), df['flex2'].max()),
                            (df['flex3'].min(), df['flex3'].max()),
                            (df['flex4'].min(), df['flex4'].max()),
                            (df['flex5'].min(), df['flex5'].max())
                        ]
                    })
                    
                except Exception as e:
                    print(f"  ❌ 분석 실패: {filename} - {e}")
        
        # 분석 결과 출력
        print("📈 데이터 분석 결과:")
        for class_name, data in class_data.items():
            print(f"\n🎯 {class_name} 클래스:")
            print(f"  📄 컬럼: {data['columns']}")
            print(f"  📊 샘플 수: {len(data['samples'])}")
            
            for sample in data['samples']:
                print(f"    📁 {sample['file']}: {sample['rows']}행")
                print(f"      📐 Pitch: {sample['pitch_range'][0]:.2f} ~ {sample['pitch_range'][1]:.2f}")
                print(f"      📐 Roll: {sample['roll_range'][0]:.2f} ~ {sample['roll_range'][1]:.2f}")
                print(f"      📐 Yaw: {sample['yaw_range'][0]:.2f} ~ {sample['yaw_range'][1]:.2f}")
                print(f"      🔧 Flex: {[f'{r[0]:.0f}~{r[1]:.0f}' for r in sample['flex_ranges']]}")

def main():
    """메인 실행 함수"""
    adapter = UnifiedDataAdapter()
    
    # 1. 데이터셋 다운로드
    if not adapter.download_unified_dataset(max_files_per_class=3):
        print("❌ 데이터셋 다운로드 실패")
        return
    
    # 2. 데이터 분석
    adapter.analyze_unified_data()
    
    # 3. KSL 형식으로 변환
    if adapter.process_all_unified_data():
        print("🎉 Unified 데이터 통합 완료!")
    else:
        print("❌ 데이터 변환 실패")

if __name__ == "__main__":
    main()

