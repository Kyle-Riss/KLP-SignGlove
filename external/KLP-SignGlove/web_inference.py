#!/usr/bin/env python3
"""
웹 기반 SignGlove 추론 시스템
Flask를 사용한 웹 인터페이스
"""

from flask import Flask, render_template, request, jsonify, Response
import torch
import numpy as np
import json
import time
import sys
from pathlib import Path
from collections import deque
from sklearn.preprocessing import StandardScaler
import threading
import queue

# 과적합 방지 모델 클래스 import
sys.path.append('.')
from anti_overfitting_gru import AntiOverfittingGRUModel

app = Flask(__name__)

class WebInferenceSystem:
    """웹 추론 시스템"""
    
    def __init__(self, model_path='best_anti_overfitting_model.pth', sequence_length=20):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = sequence_length
        self.class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 데이터 버퍼
        self.data_buffer = deque(maxlen=sequence_length)
        self.scaler = StandardScaler()
        
        # 모델 로드
        self.model = self.load_model(model_path)
        
        # 추론 설정
        self.confidence_threshold = 0.3
        self.prediction_history = deque(maxlen=50)
        
        # 실시간 처리 설정
        self.is_running = False
        self.prediction_queue = queue.Queue()
        
        print(f"🌐 웹 추론 시스템 초기화 완료 (디바이스: {self.device})")
    
    def load_model(self, model_path):
        """모델 로드"""
        model = AntiOverfittingGRUModel(
            input_size=8,
            hidden_size=32,
            num_layers=1,
            num_classes=24,
            dropout=0.0
        )
        
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"📁 모델 로드: {model_path}")
        else:
            print(f"⚠️ 모델 파일을 찾을 수 없습니다: {model_path}")
            return None
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_data(self, sensor_data):
        """센서 데이터 전처리"""
        # 데이터 정규화
        if len(self.data_buffer) == 0:
            self.scaler.fit(sensor_data.reshape(1, -1))
        
        normalized_data = self.scaler.transform(sensor_data.reshape(1, -1)).flatten()
        self.data_buffer.append(normalized_data)
        
        # 시퀀스가 충분히 쌓이면 추론 준비
        if len(self.data_buffer) >= self.sequence_length:
            sequence = np.array(list(self.data_buffer))
            return torch.FloatTensor(sequence).unsqueeze(0)
        
        return None
    
    def predict(self, sequence):
        """단일 시퀀스 예측"""
        if self.model is None:
            return None, 0.0, []
        
        with torch.no_grad():
            sequence = sequence.to(self.device)
            output = self.model(sequence)
            probabilities = torch.softmax(output, dim=1)
            
            # 모든 클래스의 확률
            all_probs = probabilities.cpu().numpy().flatten()
            
            # 최고 확률과 클래스
            max_prob, predicted_class = torch.max(probabilities, 1)
            
            return predicted_class.item(), max_prob.item(), all_probs
    
    def get_prediction_with_confidence(self, sequence):
        """신뢰도를 고려한 예측"""
        predicted_class, confidence, all_probs = self.predict(sequence)
        
        if confidence >= self.confidence_threshold:
            class_name = self.class_names[predicted_class]
            
            # 예측 히스토리에 추가
            self.prediction_history.append({
                'class': class_name,
                'confidence': confidence,
                'timestamp': time.time(),
                'all_probabilities': all_probs.tolist()
            })
            
            return class_name, confidence, all_probs
        else:
            return None, confidence, all_probs
    
    def simulate_sensor_data(self):
        """센서 데이터 시뮬레이션"""
        # 실제 센서 데이터를 시뮬레이션
        sensor_data = np.random.normal(0, 1, 8)
        return sensor_data

# 전역 추론 시스템 인스턴스
inference_system = WebInferenceSystem()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """실시간 예측 API"""
    try:
        data = request.get_json()
        
        if data is None:
            # 시뮬레이션 데이터 사용
            sensor_data = inference_system.simulate_sensor_data()
        else:
            # 실제 센서 데이터 사용
            sensor_data = np.array(data.get('sensor_data', []))
            if len(sensor_data) != 8:
                return jsonify({'error': '센서 데이터는 8개 값이어야 합니다'}), 400
        
        # 데이터 전처리
        sequence = inference_system.preprocess_data(sensor_data)
        
        if sequence is not None:
            # 예측 수행
            predicted_class, confidence, all_probs = inference_system.get_prediction_with_confidence(sequence)
            
            # 결과 준비
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probs.tolist(),
                'class_names': inference_system.class_names,
                'buffer_size': len(inference_system.data_buffer),
                'sequence_length': inference_system.sequence_length,
                'timestamp': time.time()
            }
            
            return jsonify(result)
        else:
            return jsonify({
                'status': 'buffering',
                'buffer_size': len(inference_system.data_buffer),
                'sequence_length': inference_system.sequence_length
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    """예측 히스토리 API"""
    history = list(inference_system.prediction_history)
    return jsonify({'history': history})

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """설정 API"""
    if request.method == 'POST':
        data = request.get_json()
        if 'confidence_threshold' in data:
            new_threshold = float(data['confidence_threshold'])
            if 0.0 <= new_threshold <= 1.0:
                inference_system.confidence_threshold = new_threshold
                return jsonify({'success': True, 'confidence_threshold': new_threshold})
            else:
                return jsonify({'error': '신뢰도 임계값은 0.0과 1.0 사이여야 합니다'}), 400
    
    return jsonify({
        'confidence_threshold': inference_system.confidence_threshold,
        'sequence_length': inference_system.sequence_length,
        'device': inference_system.device,
        'model_loaded': inference_system.model is not None
    })

@app.route('/api/simulate')
def simulate():
    """시뮬레이션 API"""
    sensor_data = inference_system.simulate_sensor_data()
    return jsonify({'sensor_data': sensor_data.tolist()})

# HTML 템플릿 생성
def create_templates():
    """HTML 템플릿 생성"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignGlove 실시간 추론</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .prediction-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
        }
        .prediction-result {
            text-align: center;
            font-size: 4em;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            transition: width 0.3s ease;
        }
        .controls {
            margin: 20px 0;
        }
        .btn {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            transition: background 0.3s ease;
        }
        .btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        .btn.active {
            background: #4ecdc4;
        }
        .history-panel {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        .history-item {
            background: rgba(255, 255, 255, 0.1);
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .probability-chart {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .chart-container {
            height: 200px;
            position: relative;
        }
        .status {
            text-align: center;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status.buffering {
            background: rgba(255, 193, 7, 0.3);
        }
        .status.ready {
            background: rgba(40, 167, 69, 0.3);
        }
        .status.error {
            background: rgba(220, 53, 69, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 SignGlove 실시간 추론</h1>
            <p>과적합 방지 모델을 사용한 한국어 자음/모음 인식</p>
        </div>
        
        <div class="main-content">
            <div class="prediction-panel">
                <h2>📊 실시간 예측</h2>
                <div class="status" id="status">버퍼링 중...</div>
                <div class="prediction-result" id="prediction">-</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidence-fill" style="width: 0%"></div>
                </div>
                <div id="confidence-text">신뢰도: 0%</div>
                
                <div class="controls">
                    <button class="btn" id="start-btn">시작</button>
                    <button class="btn" id="stop-btn">정지</button>
                    <button class="btn" id="clear-btn">초기화</button>
                </div>
                
                <div class="probability-chart">
                    <h3>📈 클래스별 확률</h3>
                    <div class="chart-container" id="chart-container">
                        <canvas id="probability-chart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="history-panel">
                <h2>📝 예측 히스토리</h2>
                <div id="history-list"></div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        let isRunning = false;
        let chart = null;
        
        // 차트 초기화
        function initChart() {
            const ctx = document.getElementById('probability-chart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ'],
                    datasets: [{
                        label: '확률',
                        data: Array(24).fill(0),
                        backgroundColor: 'rgba(255, 255, 255, 0.3)',
                        borderColor: 'rgba(255, 255, 255, 0.8)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ticks: {
                                color: 'white'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        },
                        x: {
                            ticks: {
                                color: 'white'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'white'
                            }
                        }
                    }
                }
            });
        }
        
        // 예측 수행
        async function performPrediction() {
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });
                
                const data = await response.json();
                
                if (data.error) {
                    updateStatus('error', data.error);
                    return;
                }
                
                if (data.status === 'buffering') {
                    updateStatus('buffering', `버퍼링 중... (${data.buffer_size}/${data.sequence_length})`);
                    return;
                }
                
                updateStatus('ready', '예측 준비 완료');
                
                if (data.predicted_class) {
                    document.getElementById('prediction').textContent = data.predicted_class;
                    document.getElementById('confidence-fill').style.width = (data.confidence * 100) + '%';
                    document.getElementById('confidence-text').textContent = `신뢰도: ${(data.confidence * 100).toFixed(1)}%`;
                    
                    // 차트 업데이트
                    if (chart && data.all_probabilities) {
                        chart.data.datasets[0].data = data.all_probabilities;
                        chart.update();
                    }
                    
                    // 히스토리 업데이트
                    updateHistory();
                }
                
            } catch (error) {
                updateStatus('error', '예측 중 오류 발생');
                console.error('Error:', error);
            }
        }
        
        // 상태 업데이트
        function updateStatus(type, message) {
            const status = document.getElementById('status');
            status.className = `status ${type}`;
            status.textContent = message;
        }
        
        // 히스토리 업데이트
        async function updateHistory() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                
                const historyList = document.getElementById('history-list');
                historyList.innerHTML = '';
                
                data.history.slice(-10).reverse().forEach(item => {
                    const historyItem = document.createElement('div');
                    historyItem.className = 'history-item';
                    historyItem.innerHTML = `
                        <span>${item.class} (${(item.confidence * 100).toFixed(1)}%)</span>
                        <span>${new Date(item.timestamp * 1000).toLocaleTimeString()}</span>
                    `;
                    historyList.appendChild(historyItem);
                });
                
            } catch (error) {
                console.error('Error updating history:', error);
            }
        }
        
        // 이벤트 리스너
        document.getElementById('start-btn').addEventListener('click', () => {
            isRunning = true;
            document.getElementById('start-btn').classList.add('active');
            document.getElementById('stop-btn').classList.remove('active');
        });
        
        document.getElementById('stop-btn').addEventListener('click', () => {
            isRunning = false;
            document.getElementById('start-btn').classList.remove('active');
            document.getElementById('stop-btn').classList.add('active');
        });
        
        document.getElementById('clear-btn').addEventListener('click', () => {
            document.getElementById('prediction').textContent = '-';
            document.getElementById('confidence-fill').style.width = '0%';
            document.getElementById('confidence-text').textContent = '신뢰도: 0%';
            if (chart) {
                chart.data.datasets[0].data = Array(24).fill(0);
                chart.update();
            }
        });
        
        // 주기적 예측
        setInterval(() => {
            if (isRunning) {
                performPrediction();
            }
        }, 1000);
        
        // 초기화
        initChart();
        updateHistory();
    </script>
</body>
</html>
    '''
    
    with open(templates_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("📁 HTML 템플릿이 생성되었습니다.")

def main():
    """메인 함수"""
    print("🌐 SignGlove 웹 추론 시스템")
    print("=" * 50)
    
    # HTML 템플릿 생성
    create_templates()
    
    # Flask 앱 실행
    print("🚀 웹 서버 시작 중...")
    print("📱 브라우저에서 http://localhost:5000 으로 접속하세요")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()



