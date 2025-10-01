"""
데이터 로딩 관련 유틸리티 함수들
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import List


def find_signglove_files(data_dir: str) -> List[str]:
    """SignGlove 데이터셋의 모든 CSV 파일을 찾습니다."""
    # 34개 클래스 (자음 14개 + 모음 10개 + 숫자 10개)
    consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    consonant_files = []
    vowel_files = []
    number_files = []
    
    # SignGlove 데이터셋 구조: datasets/{class}/{session}/episode_*.csv
    for consonant in consonants:
        consonant_pattern = os.path.join(data_dir, consonant, "*", "episode_*.csv")
        files = glob.glob(consonant_pattern)
        consonant_files.extend(files)
    
    for vowel in vowels:
        vowel_pattern = os.path.join(data_dir, vowel, "*", "episode_*.csv")
        files = glob.glob(vowel_pattern)
        vowel_files.extend(files)
    
    for number in numbers:
        number_pattern = os.path.join(data_dir, number, "*", "episode_*.csv")
        files = glob.glob(number_pattern)
        number_files.extend(files)
    
    files = consonant_files + vowel_files + number_files
    print(f"Found {len(consonant_files)} consonant files, {len(vowel_files)} vowel files, {len(number_files)} number files")
    print(f"Total: {len(files)} episode files from SignGlove dataset (34 classes)")
    return files


def extract_class_from_filename(filepath: str) -> str:
    """파일 경로에서 클래스 이름을 추출합니다."""
    # SignGlove 데이터셋 구조: datasets/{class}/{session}/episode_*.csv
    path_parts = filepath.split('/')
    for part in path_parts:
        # 한글 자모 (ㄱ-ㅎ, ㅏ-ㅣ)
        if len(part) == 1 and ord(part) >= 0x3131 and ord(part) <= 0x318E:
            return part
        # 숫자 (0-9)
        elif len(part) == 1 and part.isdigit():
            return part
    return "unknown"


def load_csv_file(filepath: str) -> np.ndarray:
    """단일 CSV 파일을 로드하고 8개 채널을 추출합니다."""
    try:
        df = pd.read_csv(filepath)
        
        # 8개 채널 추출: flex1-5, pitch, roll, yaw
        channels = ['flex1', 'flex2', 'flex3', 'flex4', 'flex5', 'pitch', 'roll', 'yaw']
        data = df[channels].values
        
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.array([])
