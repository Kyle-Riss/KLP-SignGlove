#!/usr/bin/env python3
"""
EGRU API Client Test: Test the EGRU API server
- Test all API endpoints
- Verify inference functionality
- Performance testing
"""

import requests
import json
import time
import os
from typing import Dict, Any

class EGRUAPIClient:
    """EGRU API client for testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
        print(f"🚀 EGRU API Client initialized")
        print(f"📍 Base URL: {base_url}")
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        print(f"\n🔍 Testing root endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint: {data}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        print(f"\n🔍 Testing health check...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check: {data['status']}")
                print(f"📊 Model loaded: {data['model_loaded']}")
                print(f"🖥️ Device: {data['device']}")
                print(f"📈 Best accuracy: {data['model_info'].get('best_accuracy', 'N/A')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        print(f"\n🔍 Testing model info...")
        
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Model info retrieved")
                print(f"📊 Input size: {data['model_info']['input_size']}")
                print(f"🎯 Number of classes: {data['num_classes']}")
                print(f"🇰🇷 Korean classes: {', '.join(data['korean_classes'][:10])}...")
                return True
            else:
                print(f"❌ Model info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Model info error: {e}")
            return False
    
    def test_files_list(self) -> bool:
        """Test files list endpoint"""
        print(f"\n🔍 Testing files list...")
        
        try:
            response = self.session.get(f"{self.base_url}/files/list")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Files list retrieved")
                print(f"📁 Total files: {data['total_files']}")
                print(f"📂 Directory: {data['directory']}")
                if data['files']:
                    print(f"📄 Sample files: {data['files'][:3]}")
                return True
            else:
                print(f"❌ Files list failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Files list error: {e}")
            return False
    
    def test_batch_inference_from_directory(self) -> bool:
        """Test batch inference from directory"""
        print(f"\n🔍 Testing batch inference from directory...")
        
        try:
            payload = {
                "max_files": 10,
                "confidence_threshold": 0.5
            }
            
            response = self.session.post(
                f"{self.base_url}/inference/batch-from-directory",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Batch inference completed")
                print(f"📊 Total files: {data['total_files']}")
                print(f"🎯 Accuracy: {data['accuracy']:.1%}")
                print(f"⚡ Avg inference time: {data['average_inference_time_ms']:.2f} ms")
                print(f"🕐 Total time: {data['total_processing_time_seconds']:.2f} seconds")
                
                # Class results
                print(f"\n📊 Class Results:")
                for class_name, stats in data['class_results'].items():
                    accuracy = stats['correct'] / stats['total']
                    print(f"  {class_name}: {accuracy:.1%} ({stats['correct']}/{stats['total']})")
                
                return True
            else:
                print(f"❌ Batch inference failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Batch inference error: {e}")
            return False
    
    def test_single_file_inference(self, h5_file_path: str) -> bool:
        """Test single file inference"""
        print(f"\n🔍 Testing single file inference: {os.path.basename(h5_file_path)}")
        
        try:
            if not os.path.exists(h5_file_path):
                print(f"❌ File not found: {h5_file_path}")
                return False
            
            with open(h5_file_path, 'rb') as f:
                files = {'file': (os.path.basename(h5_file_path), f, 'application/octet-stream')}
                
                response = self.session.post(
                    f"{self.base_url}/inference/single",
                    files=files,
                    params={'confidence_threshold': 0.5}
                )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Single inference completed")
                print(f"📊 True label: {data['true_label']}")
                print(f"🎯 Predicted: {data['predicted_label']}")
                print(f"✅ Correct: {data['correct']}")
                print(f"🎲 Confidence: {data['confidence']:.1%}")
                print(f"⚡ Inference time: {data['inference_time_ms']:.2f} ms")
                return True
            else:
                print(f"❌ Single inference failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Single inference error: {e}")
            return False
    
    def performance_test(self, num_files: int = 20) -> Dict[str, Any]:
        """Performance test with multiple files"""
        print(f"\n🚀 Performance test with {num_files} files...")
        
        try:
            # Get list of files
            response = self.session.get(f"{self.base_url}/files/list")
            if response.status_code != 200:
                print(f"❌ Failed to get files list for performance test")
                return {}
            
            files_data = response.json()
            available_files = files_data['files'][:num_files]
            
            if not available_files:
                print(f"❌ No files available for performance test")
                return {}
            
            print(f"📁 Testing with {len(available_files)} files...")
            
            # Test batch inference
            payload = {
                "max_files": len(available_files),
                "confidence_threshold": 0.5
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/inference/batch-from-directory",
                json=payload
            )
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                performance_results = {
                    'total_files': data['total_files'],
                    'accuracy': data['accuracy'],
                    'avg_inference_time_ms': data['average_inference_time_ms'],
                    'total_processing_time_seconds': data['total_processing_time_seconds'],
                    'api_response_time_seconds': total_time,
                    'files_per_second': data['total_files'] / data['total_processing_time_seconds'],
                    'class_results': data['class_results']
                }
                
                print(f"✅ Performance test completed")
                print(f"📊 Results:")
                print(f"  🎯 Accuracy: {performance_results['accuracy']:.1%}")
                print(f"  ⚡ Avg inference time: {performance_results['avg_inference_time_ms']:.2f} ms")
                print(f"  🚀 Files per second: {performance_results['files_per_second']:.1f}")
                print(f"  🕐 Total processing time: {performance_results['total_processing_time_seconds']:.2f} seconds")
                print(f"  🌐 API response time: {performance_results['api_response_time_seconds']:.2f} seconds")
                
                return performance_results
            else:
                print(f"❌ Performance test failed: {response.status_code}")
                return {}
                
        except Exception as e:
            print(f"❌ Performance test error: {e}")
            return {}
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests"""
        print(f"🧪 Running EGRU API tests...")
        print(f"=" * 50)
        
        test_results = {}
        
        # Basic endpoint tests
        test_results['root'] = self.test_root_endpoint()
        test_results['health'] = self.test_health_check()
        test_results['model_info'] = self.test_model_info()
        test_results['files_list'] = self.test_files_list()
        
        # Inference tests
        test_results['batch_inference'] = self.test_batch_inference_from_directory()
        
        # Performance test
        performance_results = self.performance_test(20)
        test_results['performance'] = len(performance_results) > 0
        
        # Summary
        print(f"\n📊 Test Results Summary")
        print(f"=" * 50)
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name:20}: {status}")
        
        passed = sum(test_results.values())
        total = len(test_results)
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1%})")
        
        return test_results

def main():
    """Main test function"""
    print("🚀 EGRU API Client Test")
    print("=" * 50)
    
    # Initialize client
    client = EGRUAPIClient()
    
    # Run all tests
    results = client.run_all_tests()
    
    if results.get('health', False):
        print(f"\n🎉 EGRU API is working correctly!")
        print(f"🌐 Access API documentation at: http://localhost:8000/docs")
        print(f"📊 Access ReDoc at: http://localhost:8000/redoc")
    else:
        print(f"\n❌ EGRU API has issues. Check server status.")

if __name__ == "__main__":
    main()

