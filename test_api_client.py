"""
SignGlove API 테스트 클라이언트
API 서버의 모든 엔드포인트를 테스트하는 스크립트
"""

import requests
import json
import time
import random
from typing import Dict, List, Any

class SignGloveAPIClient:
    """SignGlove API 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_root(self) -> Dict[str, Any]:
        """루트 엔드포인트 테스트"""
        print("🔍 루트 엔드포인트 테스트...")
        response = self.session.get(f"{self.base_url}/")
        print(f"  상태 코드: {response.status_code}")
        print(f"  응답: {response.json()}")
        return response.json()
    
    def test_health(self) -> Dict[str, Any]:
        """헬스 체크 테스트"""
        print("\n🔍 헬스 체크 테스트...")
        response = self.session.get(f"{self.base_url}/health")
        print(f"  상태 코드: {response.status_code}")
        data = response.json()
        print(f"  서버 상태: {data.get('status')}")
        print(f"  모델 로드: {data.get('model_loaded')}")
        if 'performance_stats' in data:
            stats = data['performance_stats']
            print(f"  FPS: {stats.get('fps', 'N/A')}")
            print(f"  평균 지연시간: {stats.get('avg_latency_ms', 'N/A')}ms")
        return data
    
    def test_model_info(self) -> Dict[str, Any]:
        """모델 정보 테스트"""
        print("\n🔍 모델 정보 테스트...")
        response = self.session.get(f"{self.base_url}/model/info")
        print(f"  상태 코드: {response.status_code}")
        data = response.json()
        print(f"  모델 이름: {data.get('model_name')}")
        print(f"  모델 버전: {data.get('model_version')}")
        print(f"  정확도: {data.get('accuracy', 0):.2%}")
        print(f"  클래스 수: {data.get('num_classes')}")
        return data
    
    def test_performance(self) -> Dict[str, Any]:
        """성능 통계 테스트"""
        print("\n🔍 성능 통계 테스트...")
        response = self.session.get(f"{self.base_url}/model/performance")
        print(f"  상태 코드: {response.status_code}")
        data = response.json()
        print(f"  FPS: {data.get('fps', 'N/A')}")
        print(f"  평균 지연시간: {data.get('avg_latency_ms', 'N/A')}ms")
        print(f"  총 예측 수: {data.get('total_predictions', 'N/A')}")
        print(f"  버퍼 사용률: {data.get('buffer_utilization', 'N/A')}")
        return data
    
    def test_classes(self) -> Dict[str, Any]:
        """지원 클래스 테스트"""
        print("\n🔍 지원 클래스 테스트...")
        response = self.session.get(f"{self.base_url}/classes")
        print(f"  상태 코드: {response.status_code}")
        data = response.json()
        print(f"  자음: {len(data.get('consonants', []))}개")
        print(f"  모음: {len(data.get('vowels', []))}개")
        print(f"  전체: {len(data.get('all_classes', []))}개")
        return data
    
    def generate_test_sensor_data(self) -> Dict[str, Any]:
        """테스트용 센서 데이터 생성"""
        return {
            "timestamp": time.time(),
            "pitch": random.uniform(-45, 45),
            "roll": random.uniform(-45, 45),
            "yaw": random.uniform(-45, 45),
            "flex1": random.randint(500, 900),
            "flex2": random.randint(500, 900),
            "flex3": random.randint(500, 900),
            "flex4": random.randint(500, 900),
            "flex5": random.randint(500, 900),
            "source": "test_client"
        }
    
    def test_single_predict(self) -> Dict[str, Any]:
        """단일 예측 테스트"""
        print("\n🔍 단일 예측 테스트...")
        sensor_data = self.generate_test_sensor_data()
        print(f"  센서 데이터: {sensor_data}")
        
        response = self.session.post(f"{self.base_url}/predict", json=sensor_data)
        print(f"  상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  예측 결과: {data.get('predicted_class')}")
            print(f"  신뢰도: {data.get('confidence', 0):.3f}")
            print(f"  안정성: {data.get('stability_score', 0):.3f}")
            print(f"  처리 시간: {data.get('processing_time_ms', 0):.2f}ms")
            return data
        else:
            print(f"  오류: {response.text}")
            return {}
    
    def test_batch_predict(self) -> List[Dict[str, Any]]:
        """배치 예측 테스트"""
        print("\n🔍 배치 예측 테스트...")
        sensor_data_list = [self.generate_test_sensor_data() for _ in range(3)]
        batch_data = {
            "sensor_data": sensor_data_list,
            "window_size": 20,
            "stride": 10
        }
        
        response = self.session.post(f"{self.base_url}/predict/batch", json=batch_data)
        print(f"  상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"  배치 크기: {len(results)}")
            for i, result in enumerate(results):
                print(f"    결과 {i+1}: {result.get('predicted_class')} (신뢰도: {result.get('confidence', 0):.3f})")
            return results
        else:
            print(f"  오류: {response.text}")
            return []
    
    def test_stable_predict(self) -> Dict[str, Any]:
        """안정적 예측 테스트"""
        print("\n🔍 안정적 예측 테스트...")
        sensor_data = self.generate_test_sensor_data()
        
        response = self.session.post(f"{self.base_url}/predict/stable", json=sensor_data)
        print(f"  상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  예측 결과: {data.get('predicted_class')}")
            print(f"  신뢰도: {data.get('confidence', 0):.3f}")
            print(f"  안정성: {data.get('stability_score', 0):.3f}")
            return data
        else:
            print(f"  오류: {response.text}")
            return {}
    
    def test_confidence_config(self) -> Dict[str, Any]:
        """신뢰도 설정 테스트"""
        print("\n🔍 신뢰도 설정 테스트...")
        threshold = 0.8
        response = self.session.post(f"{self.base_url}/config/confidence", json={"threshold": threshold})
        print(f"  상태 코드: {response.status_code}")
        data = response.json()
        print(f"  설정 결과: {data.get('message')}")
        return data
    
    def test_buffer_clear(self) -> Dict[str, Any]:
        """버퍼 초기화 테스트"""
        print("\n🔍 버퍼 초기화 테스트...")
        response = self.session.post(f"{self.base_url}/buffer/clear")
        print(f"  상태 코드: {response.status_code}")
        data = response.json()
        print(f"  초기화 결과: {data.get('message')}")
        return data
    
    def run_performance_test(self, num_requests: int = 100) -> Dict[str, Any]:
        """성능 테스트"""
        print(f"\n🚀 성능 테스트 ({num_requests}개 요청)...")
        
        start_time = time.time()
        successful_requests = 0
        total_processing_time = 0
        
        for i in range(num_requests):
            try:
                sensor_data = self.generate_test_sensor_data()
                request_start = time.time()
                
                response = self.session.post(f"{self.base_url}/predict", json=sensor_data)
                
                if response.status_code == 200:
                    successful_requests += 1
                    result = response.json()
                    processing_time = result.get('processing_time_ms', 0)
                    total_processing_time += processing_time
                
                # 진행률 표시
                if (i + 1) % 10 == 0:
                    print(f"  진행률: {i + 1}/{num_requests}")
                    
            except Exception as e:
                print(f"  요청 {i + 1} 실패: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 결과 계산
        success_rate = successful_requests / num_requests * 100
        avg_processing_time = total_processing_time / successful_requests if successful_requests > 0 else 0
        requests_per_second = successful_requests / total_time
        
        print(f"  총 시간: {total_time:.2f}초")
        print(f"  성공률: {success_rate:.1f}%")
        print(f"  평균 처리 시간: {avg_processing_time:.2f}ms")
        print(f"  초당 요청 수: {requests_per_second:.1f}")
        
        return {
            "total_time": total_time,
            "success_rate": success_rate,
            "avg_processing_time": avg_processing_time,
            "requests_per_second": requests_per_second
        }
    
    def run_full_test(self):
        """전체 API 테스트 실행"""
        print("🎯 SignGlove API 전체 테스트 시작")
        print("=" * 50)
        
        try:
            # 기본 엔드포인트 테스트
            self.test_root()
            self.test_health()
            self.test_model_info()
            self.test_performance()
            self.test_classes()
            
            # 예측 테스트
            self.test_single_predict()
            self.test_batch_predict()
            self.test_stable_predict()
            
            # 설정 테스트
            self.test_confidence_config()
            self.test_buffer_clear()
            
            # 성능 테스트
            self.run_performance_test(50)
            
            print("\n✅ 모든 테스트 완료!")
            
        except requests.exceptions.ConnectionError:
            print("❌ API 서버에 연결할 수 없습니다.")
            print("   서버가 실행 중인지 확인하세요: python server/main.py")
        except Exception as e:
            print(f"❌ 테스트 중 오류 발생: {e}")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SignGlove API 테스트 클라이언트')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='API 서버 URL')
    parser.add_argument('--performance', type=int, default=50,
                       help='성능 테스트 요청 수')
    
    args = parser.parse_args()
    
    client = SignGloveAPIClient(args.url)
    
    if args.performance > 0:
        print(f"🚀 성능 테스트만 실행 ({args.performance}개 요청)")
        client.run_performance_test(args.performance)
    else:
        client.run_full_test()

if __name__ == "__main__":
    main()
