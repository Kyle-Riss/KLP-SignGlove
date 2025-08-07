import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_collection.label_mapping import KSLLabelMapper

class KSLCsvDataset(Dataset):
    def __init__(self, csv_dir, window_size=20, stride=10, transform=None, use_labeling=True):
        self.files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.use_labeling = use_labeling
        self.data = []
        self.labels = []
        
        # 라벨 매퍼 초기화
        if self.use_labeling:
            self.label_mapper = KSLLabelMapper()
            # 현재 데이터에 있는 클래스만 사용 (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ)
            self.available_classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
            self.class_to_simplified_id = {cls: idx for idx, cls in enumerate(self.available_classes)}
        
        # 각 파일별로 데이터 처리
        for file_path in self.files:
            filename = os.path.basename(file_path)
            try:
                df = pd.read_csv(file_path, encoding='latin1')
                
                # 라벨 추출
                if self.use_labeling:
                    original_label = self.label_mapper.extract_label_from_filename(filename)
                    if original_label is None:
                        print(f"Warning: 라벨 추출 실패 - {filename}, 기본값 0 사용")
                        label = 0
                    else:
                        # 원래 클래스명을 찾아서 simplified ID로 변환
                        original_class = self.label_mapper.id_to_class.get(original_label, None)
                        if original_class in self.class_to_simplified_id:
                            label = self.class_to_simplified_id[original_class]
                        else:
                            print(f"Warning: 알 수 없는 클래스 {original_class} in {filename}, 기본값 0 사용")
                            label = 0
                else:
                    label = 0
                
                self._window_dataframe(df, label)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    def _window_dataframe(self, df, label):
        """윈도우 단위로 데이터 분할 및 라벨 할당"""
        try:
            # 컬럼명 확인 및 매핑
            available_cols = df.columns.tolist()
            flex_cols = ['flex1','flex2','flex3','flex4','flex5']
            
            # 각 방향의 컬럼을 찾기
            pitch_col = [col for col in available_cols if 'pitch' in col.lower()][0]
            roll_col = [col for col in available_cols if 'roll' in col.lower()][0]
            yaw_col = [col for col in available_cols if 'yaw' in col.lower()][0]
            
            target_cols = flex_cols + [pitch_col, roll_col, yaw_col]
            arr = df[target_cols].values
            
            for start in range(0, len(arr)-self.window_size+1, self.stride):
                window = arr[start:start+self.window_size]
                self.data.append(window)
                self.labels.append(label)  # 각 윈도우에 동일한 라벨 할당
                
        except KeyError as e:
            print(f"컬럼 오류: {e}")
            return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample.astype(np.float32), label
    
    def get_class_distribution(self):
        """클래스별 데이터 분포 반환"""
        if not self.use_labeling:
            return {"class_0": len(self.labels)}
            
        unique, counts = np.unique(self.labels, return_counts=True)
        distribution = {}
        
        for class_id, count in zip(unique, counts):
            if self.use_labeling and class_id < len(self.available_classes):
                class_name = self.available_classes[class_id]
                distribution[f"{class_name}({class_id})"] = count
            else:
                distribution[f"unknown({class_id})"] = count
                
        return distribution
    
    def print_dataset_info(self):
        """데이터셋 정보 출력"""
        print(f"\n=== KSL Dataset 정보 ===")
        print(f"총 윈도우 수: {len(self.data)}")
        print(f"윈도우 크기: {self.window_size}")
        print(f"스트라이드: {self.stride}")
        print(f"특징 차원: {self.data[0].shape if len(self.data) > 0 else 'N/A'}")
        print(f"라벨링 사용: {self.use_labeling}")
        
        if len(self.data) > 0:
            print(f"\n클래스 분포:")
            distribution = self.get_class_distribution()
            for class_name, count in distribution.items():
                print(f"  {class_name}: {count}개")
        print("=" * 30)
