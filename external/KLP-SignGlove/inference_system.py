import torch
import torch.nn as nn
import numpy as np
import time
import json
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import threading
import queue
import os

# MLP 모델 정의
class MLPModel(nn.Module):
    def __init__(self, input_size=160, hidden_sizes=[256, 128, 64], num_classes=24, dropout=0.5):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.network(x)

class SignGloveInference:
    def __init__(self, model_path='mlp_full_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MLPModel(input_size=160, hidden_sizes=[256, 128, 64], num_classes=24, dropout=0.5)
        
        # 모델 로드
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ 모델 로드 완료: {model_path}")
        else:
            print(f"⚠️  모델 파일을 찾을 수 없습니다: {model_path}")
            return
        
        self.model.to(self.device)
        self.model.eval()
        
        # 클래스 이름 정의
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 데이터 정규화를 위한 통계 (훈련 데이터에서 계산된 값)
        self.data_min = -129.167
        self.data_max = 32.167
        
        # 추론 결과 저장
        self.prediction_history = []
        self.confidence_threshold = 0.7
        
    def preprocess_data(self, sensor_data):
        """센서 데이터 전처리"""
        # 데이터 형태 변환: (seq_len, features) -> (1, seq_len, features)
        if len(sensor_data.shape) == 2:
            sensor_data = sensor_data.reshape(1, *sensor_data.shape)
        
        # 정규화
        sensor_data = (sensor_data - self.data_min) / (self.data_max - self.data_min)
        
        # 텐서 변환
        sensor_tensor = torch.FloatTensor(sensor_data).to(self.device)
        
        return sensor_tensor
    
    def predict(self, sensor_data):
        """실시간 추론"""
        try:
            # 전처리
            processed_data = self.preprocess_data(sensor_data)
            
            # 추론
            with torch.no_grad():
                outputs = self.model(processed_data)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # 결과
            predicted_class = self.class_names[predicted.item()]
            confidence_score = confidence.item()
            
            # 신뢰도가 낮으면 "Unknown" 반환
            if confidence_score < self.confidence_threshold:
                predicted_class = "Unknown"
            
            result = {
                'prediction': predicted_class,
                'confidence': confidence_score,
                'timestamp': time.time(),
                'all_probabilities': probabilities.cpu().numpy().tolist()[0]
            }
            
            # 히스토리 저장 (최근 10개)
            self.prediction_history.append(result)
            if len(self.prediction_history) > 10:
                self.prediction_history.pop(0)
            
            return result
            
        except Exception as e:
            print(f"추론 중 오류 발생: {e}")
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def get_prediction_history(self):
        """최근 추론 결과 반환"""
        return self.prediction_history
    
    def get_class_probabilities(self):
        """모든 클래스의 확률 분포 반환"""
        if not self.prediction_history:
            return {}
        
        latest = self.prediction_history[-1]
        if 'all_probabilities' in latest:
            return dict(zip(self.class_names, latest['all_probabilities']))
        return {}

# Flask 웹 서버
app = Flask(__name__)
CORS(app)

# 전역 변수
inference_system = None
data_queue = queue.Queue()

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SignGlove 실시간 인식</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .status.online { background-color: #d4edda; color: #155724; }
        .status.offline { background-color: #f8d7da; color: #721c24; }
        .prediction { font-size: 48px; text-align: center; margin: 20px 0; padding: 20px; border: 3px solid #007bff; border-radius: 10px; }
        .confidence { font-size: 24px; text-align: center; color: #666; }
        .history { margin-top: 30px; }
        .history-item { padding: 10px; margin: 5px 0; background-color: #f8f9fa; border-radius: 5px; }
        .probabilities { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 5px; margin-top: 20px; }
        .prob-item { padding: 5px; text-align: center; background-color: #e9ecef; border-radius: 3px; font-size: 12px; }
        .high-prob { background-color: #28a745; color: white; }
        .medium-prob { background-color: #ffc107; }
        .low-prob { background-color: #dc3545; color: white; }
        .controls { text-align: center; margin: 20px 0; }
        .btn { padding: 10px 20px; margin: 5px; border: none; border-radius: 5px; cursor: pointer; }
        .btn-primary { background-color: #007bff; color: white; }
        .btn-success { background-color: #28a745; color: white; }
        .btn-danger { background-color: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 SignGlove 실시간 수화 인식</h1>
            <p>MLP 모델 기반 실시간 추론 시스템</p>
        </div>
        
        <div class="status" id="status">
            <strong>상태:</strong> <span id="statusText">초기화 중...</span>
        </div>
        
        <div class="controls">
            <button class="btn btn-primary" onclick="startInference()">추론 시작</button>
            <button class="btn btn-success" onclick="simulateData()">데이터 시뮬레이션</button>
            <button class="btn btn-danger" onclick="stopInference()">추론 중지</button>
        </div>
        
        <div class="prediction" id="currentPrediction">
            대기 중...
        </div>
        
        <div class="confidence" id="confidence">
            신뢰도: 0.0%
        </div>
        
        <div class="probabilities" id="probabilities">
            <!-- 확률 분포가 여기에 표시됩니다 -->
        </div>
        
        <div class="history">
            <h3>최근 추론 기록</h3>
            <div id="historyList">
                <!-- 히스토리가 여기에 표시됩니다 -->
            </div>
        </div>
    </div>

    <script>
        let isRunning = false;
        let updateInterval;
        
        function updateStatus(status, isOnline) {
            document.getElementById('statusText').textContent = status;
            document.getElementById('status').className = 'status ' + (isOnline ? 'online' : 'offline');
        }
        
        function updatePrediction(data) {
            document.getElementById('currentPrediction').textContent = data.prediction;
            document.getElementById('confidence').textContent = `신뢰도: ${(data.confidence * 100).toFixed(1)}%`;
            
            // 확률 분포 업데이트
            if (data.all_probabilities) {
                const probContainer = document.getElementById('probabilities');
                probContainer.innerHTML = '';
                
                const classNames = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                                   'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'];
                
                classNames.forEach((className, index) => {
                    const prob = data.all_probabilities[index];
                    const probItem = document.createElement('div');
                    probItem.className = 'prob-item';
                    
                    if (prob > 0.5) probItem.className += ' high-prob';
                    else if (prob > 0.1) probItem.className += ' medium-prob';
                    else probItem.className += ' low-prob';
                    
                    probItem.textContent = `${className}: ${(prob * 100).toFixed(1)}%`;
                    probContainer.appendChild(probItem);
                });
            }
        }
        
        function updateHistory(history) {
            const historyList = document.getElementById('historyList');
            historyList.innerHTML = '';
            
            history.slice(-5).reverse().forEach(item => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                const time = new Date(item.timestamp * 1000).toLocaleTimeString();
                historyItem.innerHTML = `
                    <strong>${item.prediction}</strong> 
                    (${(item.confidence * 100).toFixed(1)}%) - ${time}
                `;
                historyList.appendChild(historyItem);
            });
        }
        
        function startInference() {
            if (isRunning) return;
            
            isRunning = true;
            updateStatus('추론 실행 중...', true);
            
            updateInterval = setInterval(async () => {
                try {
                    const response = await fetch('/get_status');
                    const data = await response.json();
                    
                    if (data.current_prediction) {
                        updatePrediction(data.current_prediction);
                    }
                    
                    if (data.history) {
                        updateHistory(data.history);
                    }
                    
                } catch (error) {
                    console.error('상태 업데이트 오류:', error);
                }
            }, 1000);
        }
        
        function stopInference() {
            if (!isRunning) return;
            
            isRunning = false;
            clearInterval(updateInterval);
            updateStatus('추론 중지됨', false);
            document.getElementById('currentPrediction').textContent = '대기 중...';
            document.getElementById('confidence').textContent = '신뢰도: 0.0%';
        }
        
        function simulateData() {
            fetch('/simulate_data', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('시뮬레이션 데이터 전송됨:', data);
                })
                .catch(error => {
                    console.error('시뮬레이션 오류:', error);
                });
        }
        
        // 페이지 로드 시 상태 확인
        window.onload = async () => {
            try {
                const response = await fetch('/get_status');
                const data = await response.json();
                updateStatus(data.status, data.is_online);
            } catch (error) {
                updateStatus('연결 오류', false);
            }
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_status')
def get_status():
    global inference_system
    
    if inference_system is None:
        return jsonify({
            'status': '모델이 로드되지 않음',
            'is_online': False,
            'current_prediction': None,
            'history': []
        })
    
    # 최근 추론 결과
    history = inference_system.get_prediction_history()
    current_prediction = history[-1] if history else None
    
    return jsonify({
        'status': '온라인',
        'is_online': True,
        'current_prediction': current_prediction,
        'history': history
    })

@app.route('/predict', methods=['POST'])
def predict():
    global inference_system
    
    if inference_system is None:
        return jsonify({'error': '모델이 로드되지 않음'}), 500
    
    try:
        data = request.get_json()
        sensor_data = np.array(data['sensor_data'])
        
        # 추론 실행
        result = inference_system.predict(sensor_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulate_data', methods=['POST'])
def simulate_data():
    """시뮬레이션 데이터 생성"""
    global inference_system
    
    if inference_system is None:
        return jsonify({'error': '모델이 로드되지 않음'}), 500
    
    try:
        # 랜덤 센서 데이터 생성 (20 시퀀스, 8 센서)
        sensor_data = np.random.uniform(-100, 30, (20, 8))
        
        # 추론 실행
        result = inference_system.predict(sensor_data)
        
        return jsonify({
            'message': '시뮬레이션 데이터 처리됨',
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    global inference_system
    
    print("🚀 SignGlove 추론 시스템 시작")
    
    # 추론 시스템 초기화
    inference_system = SignGloveInference('mlp_full_model.pth')
    
    if inference_system.model is None:
        print("❌ 모델 로드 실패. 시스템을 종료합니다.")
        return
    
    print("✅ 추론 시스템 초기화 완료")
    print("🌐 웹 서버 시작: http://localhost:5000")
    
    # Flask 서버 시작
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()

