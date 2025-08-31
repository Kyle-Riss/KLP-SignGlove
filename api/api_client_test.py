#!/usr/bin/env python3
"""
KLP-SignGlove API 클라이언트 테스트
API 서버의 기능을 테스트하는 클라이언트 스크립트
"""

import requests
import json
import time
import numpy as np
import h5py
import os
from typing import List, Dict, Any

# API 서버 설정
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "demo_token_123"  # 데모용 토큰

class APIClient:
    """API 클라이언트 클래스"""
    
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def test_health(self) -> Dict[str, Any]:
        """건강 상태 확인 테스트"""
        print("🏥 건강 상태 확인 중...")
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 서버 상태: {data['status']}")
                print(f"   모델 로드: {data['model_loaded']}")
                print(f"   GPU 사용: {data['gpu_available']}")
                print(f"   가동 시간: {data['uptime']}")
                return data
            else:
                print(f"❌ 건강 상태 확인 실패: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            return {}
    
    def test_root(self) -> Dict[str, Any]:
        """루트 엔드포인트 테스트"""
        print("\n🏠 루트 엔드포인트 테스트 중...")
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 서버 메시지: {data['message']}")
                print(f"   버전: {data['version']}")
                return data
            else:
                print(f"❌ 루트 엔드포인트 실패: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            return {}
    
    def test_class_info(self) -> Dict[str, Any]:
        """클래스 정보 테스트"""
        print("\n📚 클래스 정보 테스트 중...")
        try:
            response = requests.get(f"{self.base_url}/class-info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 총 클래스 수: {data['total_classes']}")
                print(f"   자음: {', '.join(data['categories']['consonants'])}")
                print(f"   모음: {', '.join(data['categories']['vowels'])}")
                return data
            else:
                print(f"❌ 클래스 정보 실패: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            return {}
    
    def test_models(self) -> List[Dict[str, Any]]:
        """모델 목록 테스트"""
        print("\n🤖 모델 목록 테스트 중...")
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                models = response.json()
                print(f"✅ 사용 가능한 모델: {len(models)}개")
                for model in models:
                    print(f"   - {model['name']} ({model['type']}) - {model['status']}")
                return models
            else:
                print(f"❌ 모델 목록 실패: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            return []
    
    def test_performance(self) -> Dict[str, Any]:
        """성능 통계 테스트"""
        print("\n📊 성능 통계 테스트 중...")
        try:
            response = requests.get(f"{self.base_url}/performance")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 총 요청: {data['total_requests']}")
                print(f"   성공한 예측: {data['successful_predictions']}")
                print(f"   평균 처리 시간: {data['average_processing_time']:.3f}초")
                print(f"   정확도: {data['accuracy_rate']:.3f}")
                return data
            else:
                print(f"❌ 성능 통계 실패: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            return {}
    
    def test_prediction(self, sensor_data: List[List[float]]) -> Dict[str, Any]:
        """수화 인식 추론 테스트"""
        print("\n🎯 수화 인식 추론 테스트 중...")
        
        request_data = {
            "sensor_data": sensor_data
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/predict",
                headers=self.headers,
                json=request_data
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 예측 성공!")
                print(f"   예측 클래스: {data['predicted_class']}")
                print(f"   신뢰도: {data['confidence']:.3f}")
                print(f"   처리 시간: {data['processing_time']:.3f}초")
                print(f"   API 응답 시간: {request_time:.3f}초")
                return data
            else:
                print(f"❌ 추론 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return {}
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            return {}
    
    def test_batch_prediction(self, sensor_data_list: List[List[List[float]]]) -> Dict[str, Any]:
        """배치 추론 테스트"""
        print(f"\n📦 배치 추론 테스트 중... ({len(sensor_data_list)}개)")
        
        requests_data = [
            {"sensor_data": data} for data in sensor_data_list
        ]
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/batch-predict",
                headers=self.headers,
                json=requests_data
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 배치 추론 성공!")
                print(f"   총 요청: {data['total_requests']}")
                print(f"   성공: {data['successful']}")
                print(f"   API 응답 시간: {request_time:.3f}초")
                
                # 개별 결과 요약
                for i, result in enumerate(data['results'][:3]):  # 처음 3개만 표시
                    if result['success']:
                        pred = result['result']['predicted_class']
                        conf = result['result']['confidence']
                        print(f"   [{i}] {pred} (신뢰도: {conf:.3f})")
                    else:
                        print(f"   [{i}] 실패: {result['error']}")
                
                return data
            else:
                print(f"❌ 배치 추론 실패: {response.status_code}")
                print(f"   오류: {response.text}")
                return {}
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
            return {}

def load_test_data() -> List[List[List[float]]]:
    """테스트용 데이터 로드"""
    print("📁 테스트 데이터 로드 중...")
    
    test_data = []
    data_dir = '../real_data_filtered'
    
    # 몇 개 클래스에서 데이터 샘플 로드
    test_classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㅁ', 'ㅂ']
    
    for class_name in test_classes:
        class_dir = os.path.join(data_dir, class_name, '1')
        if os.path.exists(class_dir):
            csv_files = [f for f in os.listdir(class_dir) if f.endswith('.csv')]
            if csv_files:
                file_path = os.path.join(class_dir, csv_files[0])
                try:
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    data = df.iloc[:, :8].values.tolist()
                    test_data.append(data)
                    print(f"   ✅ {class_name}: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"   ❌ {class_name}: {e}")
    
    print(f"📊 총 {len(test_data)}개 테스트 데이터 로드 완료")
    return test_data

def main():
    """메인 테스트 함수"""
    print("🚀 KLP-SignGlove API 클라이언트 테스트 시작")
    print("=" * 50)
    
    # API 클라이언트 생성
    client = APIClient(API_BASE_URL, API_TOKEN)
    
    # 기본 기능 테스트
    print("🔍 기본 기능 테스트")
    print("-" * 30)
    
    # 1. 건강 상태 확인
    health_data = client.test_health()
    if not health_data:
        print("❌ 서버가 실행되지 않았습니다. 먼저 API 서버를 시작해주세요.")
        return
    
    # 2. 루트 엔드포인트
    client.test_root()
    
    # 3. 클래스 정보
    client.test_class_info()
    
    # 4. 모델 목록
    client.test_models()
    
    # 5. 성능 통계
    client.test_performance()
    
    # 추론 테스트
    print("\n🎯 추론 기능 테스트")
    print("-" * 30)
    
    # 테스트 데이터 로드
    test_data = load_test_data()
    
    if not test_data:
        print("❌ 테스트 데이터를 로드할 수 없습니다.")
        return
    
    # 6. 단일 추론 테스트
    print(f"\n📝 첫 번째 데이터로 단일 추론 테스트")
    client.test_prediction(test_data[0])
    
    # 7. 배치 추론 테스트
    print(f"\n📦 배치 추론 테스트")
    client.test_batch_prediction(test_data[:3])  # 처음 3개만
    
    # 최종 성능 확인
    print("\n📊 최종 성능 확인")
    print("-" * 30)
    client.test_performance()
    
    print("\n🎉 API 클라이언트 테스트 완료!")
    print("=" * 50)

if __name__ == "__main__":
    main()
