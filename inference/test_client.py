import requests
import json
import time
import numpy as np
from typing import List, Dict

class SignGloveClient:
    """SignGlove API 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """헬스 체크"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_system_info(self) -> Dict:
        """시스템 정보 조회"""
        response = self.session.get(f"{self.base_url}/info")
        return response.json()
    
    def get_labels(self) -> Dict:
        """라벨 목록 조회"""
        response = self.session.get(f"{self.base_url}/labels")
        return response.json()
    
    def predict_single(self, sensor_data: List[List[float]]) -> Dict:
        """단일 예측"""
        # 센서 데이터를 API 형식으로 변환
        api_data = []
        for data in sensor_data:
            api_data.append({
                "flex1": data[0],
                "flex2": data[1],
                "flex3": data[2],
                "flex4": data[3],
                "flex5": data[4],
                "pitch": data[5],
                "roll": data[6],
                "yaw": data[7]
            })
        
        payload = {"sensor_data": api_data}
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        return response.json()
    
    def predict_batch(self, sensor_data_list: List[List[List[float]]]) -> Dict:
        """배치 예측"""
        # 센서 데이터를 API 형식으로 변환
        api_data = []
        for sensor_data in sensor_data_list:
            for data in sensor_data:
                api_data.append({
                    "flex1": data[0],
                    "flex2": data[1],
                    "flex3": data[2],
                    "flex4": data[3],
                    "flex5": data[4],
                    "pitch": data[5],
                    "roll": data[6],
                    "yaw": data[7]
                })
        
        payload = {"sensor_data": api_data}
        response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
        return response.json()
    
    def start_realtime(self) -> Dict:
        """실시간 예측 시작"""
        response = self.session.post(f"{self.base_url}/realtime/start")
        return response.json()
    
    def add_sensor_data(self, sensor_data: List[float]) -> Dict:
        """실시간 센서 데이터 추가"""
        payload = {
            "flex1": sensor_data[0],
            "flex2": sensor_data[1],
            "flex3": sensor_data[2],
            "flex4": sensor_data[3],
            "flex5": sensor_data[4],
            "pitch": sensor_data[5],
            "roll": sensor_data[6],
            "yaw": sensor_data[7]
        }
        response = self.session.post(f"{self.base_url}/realtime/add", json=payload)
        return response.json()
    
    def reset_realtime(self) -> Dict:
        """실시간 버퍼 초기화"""
        response = self.session.post(f"{self.base_url}/realtime/reset")
        return response.json()

def generate_test_data(sequence_length: int = 20) -> List[List[float]]:
    """테스트용 센서 데이터 생성"""
    sensor_data = []
    for _ in range(sequence_length):
        # 8축 센서 데이터 시뮬레이션
        data = np.random.rand(8).tolist()
        sensor_data.append(data)
    return sensor_data

def test_single_prediction(client: SignGloveClient):
    """단일 예측 테스트"""
    print("=== 단일 예측 테스트 ===")
    
    # 테스트 데이터 생성
    sensor_data = generate_test_data(20)
    
    # 예측 수행
    start_time = time.time()
    result = client.predict_single(sensor_data)
    end_time = time.time()
    
    print(f"예측 결과: {result['predicted_label']}")
    print(f"신뢰도: {result['confidence']:.3f}")
    print(f"처리 시간: {(end_time - start_time) * 1000:.2f}ms")
    print(f"타임스탬프: {result['timestamp']}")
    
    # 상위 3개 확률 출력
    probabilities = result['probabilities']
    top_3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
    print("상위 3개 확률:")
    for label, prob in top_3:
        print(f"  {label}: {prob:.3f}")

def test_batch_prediction(client: SignGloveClient):
    """배치 예측 테스트"""
    print("\n=== 배치 예측 테스트 ===")
    
    # 테스트 데이터 생성 (3개 시퀀스)
    sensor_data_list = [generate_test_data(20) for _ in range(3)]
    
    # 배치 예측 수행
    start_time = time.time()
    result = client.predict_batch(sensor_data_list)
    end_time = time.time()
    
    print(f"총 예측 수: {result['total_predictions']}")
    print(f"처리 시간: {(end_time - start_time) * 1000:.2f}ms")
    
    # 각 예측 결과 출력
    for i, pred in enumerate(result['results']):
        print(f"예측 {i+1}: {pred['predicted_label']} (신뢰도: {pred['confidence']:.3f})")

def test_realtime_prediction(client: SignGloveClient):
    """실시간 예측 테스트"""
    print("\n=== 실시간 예측 테스트 ===")
    
    # 실시간 예측 시작
    start_result = client.start_realtime()
    print(f"실시간 예측 시작: {start_result['message']}")
    
    # 센서 데이터를 순차적으로 추가
    for i in range(25):  # 버퍼 크기보다 많은 데이터
        sensor_data = np.random.rand(8).tolist()
        result = client.add_sensor_data(sensor_data)
        
        if result['prediction']:
            pred = result['prediction']
            print(f"데이터 {i+1}: {pred['predicted_label']} (신뢰도: {pred['confidence']:.3f})")
        else:
            print(f"데이터 {i+1}: 버퍼 채우는 중... ({result['buffer_size']}/20)")
        
        time.sleep(0.1)  # 100ms 간격
    
    # 버퍼 초기화
    reset_result = client.reset_realtime()
    print(f"버퍼 초기화: {reset_result['message']}")

def main():
    """메인 함수"""
    client = SignGloveClient()
    
    try:
        # 서버 연결 확인
        print("=== SignGlove API 클라이언트 테스트 ===")
        
        health = client.health_check()
        print(f"서버 상태: {health['status']}")
        print(f"모델 로드: {health['model_loaded']}")
        
        # 시스템 정보 조회
        info = client.get_system_info()
        print(f"모델 경로: {info['model_path']}")
        print(f"디바이스: {info['device']}")
        print(f"총 파라미터 수: {info['total_parameters']:,}")
        print(f"라벨 수: {len(info['labels'])}")
        
        # 라벨 목록 조회
        labels = client.get_labels()
        print(f"사용 가능한 라벨: {labels['labels']}")
        
        # 테스트 실행
        test_single_prediction(client)
        test_batch_prediction(client)
        test_realtime_prediction(client)
        
        print("\n=== 모든 테스트 완료 ===")
        
    except requests.exceptions.ConnectionError:
        print("오류: API 서버에 연결할 수 없습니다.")
        print("서버가 실행 중인지 확인하세요: python api_server.py")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()



