"""
Madgwick 필터 데이터 어댑터
GitHub의 Madgwick IMU 데이터를 현재 프로젝트 형식으로 변환
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os

class MadgwickDataAdapter:
    def __init__(self):
        """Madgwick 데이터 어댑터 초기화"""
        self.flex_default_values = [800, 820, 810, 830, 850]  # 기본 Flex 센서 값
        
    def convert_madgwick_to_ksl_format(self, madgwick_file: str, 
                                     output_file: Optional[str] = None,
                                     add_synthetic_flex: bool = True) -> pd.DataFrame:
        """
        Madgwick 데이터를 KSL 프로젝트 형식으로 변환
        
        Args:
            madgwick_file: Madgwick CSV 파일 경로
            output_file: 출력 파일 경로 (None이면 저장 안함)
            add_synthetic_flex: 합성 Flex 센서 데이터 추가 여부
            
        Returns:
            변환된 DataFrame
        """
        try:
            # Madgwick 데이터 로드 (인코딩 시도)
            try:
                df_madgwick = pd.read_csv(madgwick_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df_madgwick = pd.read_csv(madgwick_file, encoding='latin1')
                    print("latin1 인코딩으로 로드됨")
                except UnicodeDecodeError:
                    df_madgwick = pd.read_csv(madgwick_file, encoding='cp949')
                    print("cp949 인코딩으로 로드됨")
            
            print(f"Madgwick 데이터 로드: {len(df_madgwick)}행")
            print(f"컬럼: {list(df_madgwick.columns)}")
            
            # 필요한 컬럼 확인
            required_cols = ['timestamp(ms)', 'pitch(°)', 'roll(°)', 'yaw(°)']
            missing_cols = [col for col in required_cols if col not in df_madgwick.columns]
            
            if missing_cols:
                print(f"Warning: 누락된 컬럼들: {missing_cols}")
                # 대체 컬럼 이름 시도
                col_mapping = {
                    'timestamp(ms)': ['timestamp', 'time', 'ms'],
                    'pitch(°)': ['pitch', 'pitch(°)', 'pitch(deg)', 'pitch(¡Æ)'],
                    'roll(°)': ['roll', 'roll(°)', 'roll(deg)', 'roll(¡Æ)'],
                    'yaw(°)': ['yaw', 'yaw(°)', 'yaw(deg)', 'yaw(¡Æ)']
                }
                
                for target_col, alternatives in col_mapping.items():
                    if target_col not in df_madgwick.columns:
                        for alt in alternatives:
                            if alt in df_madgwick.columns:
                                df_madgwick = df_madgwick.rename(columns={alt: target_col})
                                print(f"컬럼 매핑: {alt} -> {target_col}")
                                break
            
            # KSL 형식 DataFrame 생성
            df_ksl = pd.DataFrame()
            
            # 타임스탬프 복사
            if 'timestamp(ms)' in df_madgwick.columns:
                df_ksl['timestamp(ms)'] = df_madgwick['timestamp(ms)']
            else:
                # 타임스탬프가 없으면 생성 (20ms 간격 가정)
                df_ksl['timestamp(ms)'] = range(0, len(df_madgwick) * 20, 20)
                
            # 오일러각 복사
            for angle in ['pitch(°)', 'roll(°)', 'yaw(°)']:
                if angle in df_madgwick.columns:
                    df_ksl[angle] = df_madgwick[angle]
                else:
                    print(f"Warning: {angle} 컬럼이 없어서 0으로 설정")
                    df_ksl[angle] = 0.0
            
            # Flex 센서 데이터 추가
            if add_synthetic_flex:
                df_ksl = self._add_synthetic_flex_data(df_ksl, df_madgwick)
            else:
                # 기본값으로 설정
                for i, flex_val in enumerate(self.flex_default_values, 1):
                    df_ksl[f'flex{i}'] = flex_val
            
            print(f"변환 완료: {len(df_ksl)}행, {len(df_ksl.columns)}컬럼")
            print(f"최종 컬럼: {list(df_ksl.columns)}")
            
            # 파일 저장
            if output_file:
                df_ksl.to_csv(output_file, index=False)
                print(f"변환된 데이터 저장: {output_file}")
                
            return df_ksl
            
        except Exception as e:
            print(f"변환 중 오류: {e}")
            raise
    
    def _add_synthetic_flex_data(self, df_ksl: pd.DataFrame, df_madgwick: pd.DataFrame) -> pd.DataFrame:
        """
        IMU 데이터를 기반으로 합성 Flex 센서 데이터 생성
        
        Args:
            df_ksl: 기본 KSL DataFrame
            df_madgwick: 원본 Madgwick DataFrame
            
        Returns:
            Flex 데이터가 추가된 DataFrame
        """
        try:
            # 가속도 데이터가 있는 경우 이를 활용
            if all(col in df_madgwick.columns for col in ['ax(g)', 'ay(g)', 'az(g)']):
                ax = df_madgwick['ax(g)'].values
                ay = df_madgwick['ay(g)'].values
                az = df_madgwick['az(g)'].values
                
                # 가속도 크기 계산
                accel_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
                
                # 각 손가락별로 다른 패턴으로 Flex 값 생성
                # 엄지 (flex1): pitch 기반
                df_ksl['flex1'] = self.flex_default_values[0] + \
                                df_ksl['pitch(°)'] * 2 + np.random.normal(0, 5, len(df_ksl))
                
                # 검지 (flex2): roll 기반  
                df_ksl['flex2'] = self.flex_default_values[1] + \
                                df_ksl['roll(°)'] * 1.5 + np.random.normal(0, 5, len(df_ksl))
                
                # 중지 (flex3): yaw 기반
                df_ksl['flex3'] = self.flex_default_values[2] + \
                                df_ksl['yaw(°)'] * 1 + np.random.normal(0, 5, len(df_ksl))
                
                # 약지 (flex4): 가속도 크기 기반
                df_ksl['flex4'] = self.flex_default_values[3] + \
                                (accel_magnitude - 1) * 20 + np.random.normal(0, 5, len(df_ksl))
                
                # 소지 (flex5): 복합 패턴
                df_ksl['flex5'] = self.flex_default_values[4] + \
                                (df_ksl['pitch(°)'] + df_ksl['roll(°)']) * 0.5 + \
                                np.random.normal(0, 5, len(df_ksl))
                
                print("가속도 데이터 기반 합성 Flex 센서 데이터 생성")
                
            else:
                # 가속도 데이터가 없으면 오일러각만 사용
                # 각 손가락별로 다른 패턴으로 Flex 값 생성
                df_ksl['flex1'] = self.flex_default_values[0] + \
                                df_ksl['pitch(°)'] * 2 + np.random.normal(0, 10, len(df_ksl))
                
                df_ksl['flex2'] = self.flex_default_values[1] + \
                                df_ksl['roll(°)'] * 1.5 + np.random.normal(0, 10, len(df_ksl))
                
                df_ksl['flex3'] = self.flex_default_values[2] + \
                                df_ksl['yaw(°)'] * 1 + np.random.normal(0, 10, len(df_ksl))
                
                df_ksl['flex4'] = self.flex_default_values[3] + \
                                (df_ksl['pitch(°)'] - df_ksl['roll(°)']) * 0.8 + \
                                np.random.normal(0, 10, len(df_ksl))
                
                df_ksl['flex5'] = self.flex_default_values[4] + \
                                (df_ksl['pitch(°)'] + df_ksl['yaw(°)']) * 0.6 + \
                                np.random.normal(0, 10, len(df_ksl))
                
                print("오일러각 기반 합성 Flex 센서 데이터 생성")
            
            # Flex 값들을 합리적인 범위로 제한
            for i in range(1, 6):
                flex_col = f'flex{i}'
                df_ksl[flex_col] = np.clip(df_ksl[flex_col], 700, 900)
                
            return df_ksl
            
        except Exception as e:
            print(f"합성 Flex 데이터 생성 중 오류: {e}")
            # 오류 시 기본값 사용
            for i, flex_val in enumerate(self.flex_default_values, 1):
                df_ksl[f'flex{i}'] = flex_val + np.random.normal(0, 10, len(df_ksl))
            return df_ksl
    
    def compare_data_characteristics(self, original_file: str, converted_file: str) -> dict:
        """
        원본과 변환된 데이터의 특성 비교
        
        Args:
            original_file: 원본 Madgwick 파일
            converted_file: 변환된 KSL 파일
            
        Returns:
            비교 결과 딕셔너리
        """
        try:
            # 인코딩 처리하여 원본 파일 로드
            try:
                df_original = pd.read_csv(original_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df_original = pd.read_csv(original_file, encoding='latin1')
                except UnicodeDecodeError:
                    df_original = pd.read_csv(original_file, encoding='cp949')
            
            df_converted = pd.read_csv(converted_file)
            
            comparison = {
                'original': {
                    'rows': len(df_original),
                    'columns': list(df_original.columns),
                    'duration_ms': df_original['timestamp(ms)'].max() - df_original['timestamp(ms)'].min() if 'timestamp(ms)' in df_original.columns else 'N/A',
                    'sampling_rate': self._estimate_sampling_rate(df_original['timestamp(ms)']) if 'timestamp(ms)' in df_original.columns else 'N/A'
                },
                'converted': {
                    'rows': len(df_converted),
                    'columns': list(df_converted.columns),
                    'duration_ms': df_converted['timestamp(ms)'].max() - df_converted['timestamp(ms)'].min(),
                    'sampling_rate': self._estimate_sampling_rate(df_converted['timestamp(ms)'])
                }
            }
            
            # 공통 컬럼 통계
            common_angles = ['pitch(°)', 'roll(°)', 'yaw(°)']
            comparison['angle_stats'] = {}
            
            for angle in common_angles:
                if angle in df_original.columns and angle in df_converted.columns:
                    comparison['angle_stats'][angle] = {
                        'original_range': (df_original[angle].min(), df_original[angle].max()),
                        'converted_range': (df_converted[angle].min(), df_converted[angle].max()),
                        'original_std': df_original[angle].std(),
                        'converted_std': df_converted[angle].std()
                    }
            
            return comparison
            
        except Exception as e:
            print(f"데이터 비교 중 오류: {e}")
            return {}
    
    def _estimate_sampling_rate(self, timestamp_series: pd.Series) -> float:
        """타임스탬프에서 샘플링 레이트 추정"""
        try:
            time_diffs = timestamp_series.diff().dropna()
            avg_interval_ms = time_diffs.mean()
            return 1000.0 / avg_interval_ms if avg_interval_ms > 0 else 0
        except:
            return 0

# 사용 예시 및 테스트
if __name__ == "__main__":
    adapter = MadgwickDataAdapter()
    
    # 테스트용 Madgwick 데이터 변환
    madgwick_file = "temp_madgwick.csv"  # 다운로드된 파일
    output_file = "madgwick_converted_to_ksl.csv"
    
    if os.path.exists(madgwick_file):
        print("=== Madgwick 데이터 변환 테스트 ===")
        
        # 변환 실행
        df_converted = adapter.convert_madgwick_to_ksl_format(
            madgwick_file, output_file, add_synthetic_flex=True
        )
        
        print(f"\n변환 결과:")
        print(f"- 행 수: {len(df_converted)}")
        print(f"- 컬럼: {list(df_converted.columns)}")
        print(f"- 첫 5행:")
        print(df_converted.head())
        
        # 데이터 특성 비교
        if os.path.exists(output_file):
            comparison = adapter.compare_data_characteristics(madgwick_file, output_file)
            print(f"\n=== 데이터 특성 비교 ===")
            print(f"원본: {comparison.get('original', {})}")
            print(f"변환: {comparison.get('converted', {})}")
            
    else:
        print(f"테스트 파일 {madgwick_file}이 없습니다.")
