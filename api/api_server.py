#!/usr/bin/env python3
"""
KLP-SignGlove API Server
한국 수화 인식을 위한 REST API 서버
"""

import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from sklearn.preprocessing import StandardScaler

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="KLP-SignGlove API",
    description="한국 수화 인식을 위한 REST API 서버",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 보안 설정
security = HTTPBearer()

# 데이터 모델 정의
class PredictionRequest(BaseModel):
    """추론 요청 데이터 모델"""
    sensor_data: List[List[float]] = Field(..., description="센서 데이터 (300x8)")
    class_names: Optional[List[str]] = Field(None, description="클래스 이름 (기본값: 한국어 자음/모음)")

class PredictionResponse(BaseModel):
    """추론 응답 데이터 모델"""
    predicted_class: str = Field(..., description="예측된 클래스")
    confidence: float = Field(..., description="예측 신뢰도")
    processing_time: float = Field(..., description="처리 시간 (초)")
    timestamp: str = Field(..., description="추론 시간")

class HealthResponse(BaseModel):
    """건강 상태 응답 모델"""
    status: str = Field(..., description="서버 상태")
    model_loaded: bool = Field(..., description="모델 로드 상태")
    gpu_available: bool = Field(..., description="GPU 사용 가능 여부")
    uptime: str = Field(..., description="서버 가동 시간")

class PerformanceStats(BaseModel):
    """성능 통계 모델"""
    total_requests: int = Field(..., description="총 요청 수")
    successful_predictions: int = Field(..., description="성공한 예측 수")
    average_processing_time: float = Field(..., description="평균 처리 시간")
    accuracy_rate: float = Field(..., description="정확도")

# 전역 변수
model = None
preprocessor = None
device = None
class_names = None
start_time = None
stats = {
    "total_requests": 0,
    "successful_predictions": 0,
    "processing_times": []
}

# RGRU 클래스 (실제 훈련된 모델 구조)
class RGRU(nn.Module):
    """정규화가 강화된 모델"""
    
    def __init__(self, input_size=8, hidden_size=96, num_classes=24, dropout=0.5):
        super(RGRU, self).__init__()
        
        self.hidden_size = hidden_size
        
        # 1. 입력 정규화
        self.input_norm = nn.LayerNorm(input_size)
        
        # 2. 특징 추출
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )
        
        # 3. 시퀀스 모델링
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        
        # 4. 어텐션
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 5. 분류 헤드 (강화된 정규화)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 2),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size // 4),
            
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # 6. 가중치 정규화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화 및 정규화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 1. 입력 정규화
        x = self.input_norm(x)
        
        # 2. 특징 추출
        features = self.feature_extractor(x)
        
        # 3. GRU 처리
        gru_out, _ = self.gru(features)
        
        # 4. 어텐션 가중치
        attention_weights = self.attention(gru_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 5. 어텐션 적용
        context = torch.sum(attention_weights * gru_out, dim=1)
        
        # 6. 분류
        output = self.classifier(context)
        
        return output

# AdvancedPreprocessor 클래스
class AdvancedPreprocessor:
    """고급 전처리기"""
    def __init__(self):
        self.flex_scalers = [StandardScaler() for _ in range(5)]
        self.orientation_scalers = [StandardScaler() for _ in range(3)]
        self.is_fitted = False
    
    def fit(self, data_list):
        """전처리 파라미터 학습"""
        logger.info('🔧 전처리 파라미터 학습 중...')
        
        for sensor_idx in range(5):
            all_values = []
            for data in data_list:
                sensor_values = data[:, sensor_idx]
                valid_values = sensor_values[sensor_values > 0]
                all_values.extend(valid_values)
            
            if all_values:
                self.flex_scalers[sensor_idx].fit(np.array(all_values).reshape(-1, 1))
        
        for sensor_idx in range(3):
            all_values = []
            for data in data_list:
                sensor_values = data[:, sensor_idx + 5]
                all_values.extend(sensor_values)
            
            if all_values:
                self.orientation_scalers[sensor_idx].fit(np.array(all_values).reshape(-1, 1))
        
        self.is_fitted = True
        logger.info('✅ 전처리 파라미터 학습 완료')
    
    def transform_single(self, data):
        """단일 데이터 변환"""
        if not self.is_fitted:
            raise ValueError("전처리 파라미터를 먼저 학습해야 합니다.")
        
        processed = data.copy()
        
        # Flex 센서 처리 (0값 처리 + 정규화)
        for sensor_idx in range(5):
            sensor_values = data[:, sensor_idx]
            mean_val = np.mean(sensor_values[sensor_values > 0])
            if np.isnan(mean_val):
                mean_val = 500
            
            sensor_values[sensor_values == 0] = mean_val
            sensor_values_normalized = self.flex_scalers[sensor_idx].transform(
                sensor_values.reshape(-1, 1)
            ).flatten()
            processed[:, sensor_idx] = sensor_values_normalized
        
        # Orientation 센서 처리 (정규화)
        for sensor_idx in range(3):
            sensor_values = data[:, sensor_idx + 5]
            sensor_values_normalized = self.orientation_scalers[sensor_idx].transform(
                sensor_values.reshape(-1, 1)
            ).flatten()
            processed[:, sensor_idx + 5] = sensor_values_normalized
        
        return processed

# 모델 및 전처리기 로드
def load_model_and_preprocessor():
    """모델과 전처리기 로드"""
    global model, preprocessor, device, class_names
    
    try:
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'🔧 디바이스: {device}')
        
        # 클래스 이름 설정
        class_names = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 
                      'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
        
        # 모델 로드
        logger.info('🤖 모델 로드 중...')
        model = RGRU(input_size=8, hidden_size=96, num_classes=24, dropout=0.5)
        
        model_path = '../models/improved_rgru_model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logger.info('✅ 모델 로드 완료')
        
        # 전처리기 로드 및 학습
        logger.info('🔧 전처리기 로드 중...')
        preprocessor = AdvancedPreprocessor()
        
        # 데이터로 전처리기 학습
        all_data = []
        data_dir = '../real_data_filtered'
        
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for session in ['1', '2', '3', '4', '5']:
                    session_dir = os.path.join(class_dir, session)
                    if os.path.exists(session_dir):
                        for file in os.listdir(session_dir):
                            if file.endswith('.csv'):
                                file_path = os.path.join(session_dir, file)
                                try:
                                    df = pd.read_csv(file_path)
                                    data = df.iloc[:, :8].values
                                    all_data.append(data)
                                except:
                                    continue
        
        if all_data:
            preprocessor.fit(all_data)
            logger.info('✅ 전처리기 로드 완료')
        else:
            logger.warning('⚠️ 전처리기 학습 데이터를 찾을 수 없습니다.')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ 모델 로드 실패: {e}')
        return False

# 인증 함수
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """토큰 검증 (간단한 예시)"""
    # 실제 구현에서는 JWT 토큰 검증 로직 추가
    token = credentials.credentials
    if token == "demo_token_123":  # 데모용 토큰
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid token",
        headers={"WWW-Authenticate": "Bearer"},
    )

# API 엔드포인트

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    global start_time
    start_time = datetime.now()
    
    # 모델 로드
    if not load_model_and_preprocessor():
        logger.error("❌ 서버 시작 실패: 모델을 로드할 수 없습니다.")
        raise RuntimeError("모델 로드 실패")

@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "KLP-SignGlove API 서버에 오신 것을 환영합니다!",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """건강 상태 확인"""
    global start_time, model, device
    
    uptime = str(datetime.now() - start_time) if start_time else "Unknown"
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        gpu_available=torch.cuda.is_available(),
        uptime=uptime
    )

@app.get("/class-info", response_model=Dict[str, Any])
async def get_class_info():
    """클래스 정보 조회"""
    global class_names
    
    consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    return {
        "total_classes": len(class_names),
        "categories": {
            "consonants": consonants,
            "vowels": vowels
        }
    }

@app.get("/models", response_model=List[Dict[str, str]])
async def get_models():
    """모델 목록 조회"""
    global model
    
    models = []
    if model is not None:
        models.append({
            "name": "improved_preprocessing_model",
            "type": "RGRU",
            "status": "loaded"
        })
    
    return models

@app.get("/performance", response_model=PerformanceStats)
async def get_performance_stats():
    """성능 통계 조회"""
    global stats
    
    avg_time = np.mean(stats["processing_times"]) if stats["processing_times"] else 0.0
    accuracy = stats["successful_predictions"] / stats["total_requests"] if stats["total_requests"] > 0 else 0.0
    
    return PerformanceStats(
        total_requests=stats["total_requests"],
        successful_predictions=stats["successful_predictions"],
        average_processing_time=avg_time,
        accuracy_rate=accuracy
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, _: bool = Depends(verify_token)):
    """단일 추론"""
    global model, preprocessor, class_names, stats
    
    try:
        start_time = time.time()
        
        # 입력 데이터 전처리
        sensor_data = np.array(request.sensor_data)
        if sensor_data.shape[1] != 8:
            raise ValueError("센서 데이터는 8개 채널이어야 합니다.")
        
        # 시퀀스 길이 통일 (패딩)
        if len(sensor_data) < 300:
            padding = np.tile(sensor_data[-1:], (300 - len(sensor_data), 1))
            sensor_data = np.vstack([sensor_data, padding])
        else:
            sensor_data = sensor_data[:300]
        
        # 전처리 적용
        if preprocessor and preprocessor.is_fitted:
            sensor_data = preprocessor.transform_single(sensor_data)
        
        # 모델 추론
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sensor_data).unsqueeze(0).to(device)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        processing_time = time.time() - start_time
        
        # 통계 업데이트
        stats["total_requests"] += 1
        stats["successful_predictions"] += 1
        stats["processing_times"].append(processing_time)
        
        return PredictionResponse(
            predicted_class=class_names[predicted_idx],
            confidence=confidence,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ 추론 오류: {e}")
        stats["total_requests"] += 1
        raise HTTPException(status_code=500, detail=f"추론 중 오류가 발생했습니다: {str(e)}")

@app.post("/batch-predict", response_model=Dict[str, Any])
async def batch_predict(requests: List[PredictionRequest], _: bool = Depends(verify_token)):
    """배치 추론"""
    global stats
    
    results = []
    successful = 0
    
    for i, request in enumerate(requests):
        try:
            # 단일 추론 재사용
            result = await predict(request, _)
            results.append({
                "index": i,
                "success": True,
                "result": {
                    "predicted_class": result.predicted_class,
                    "confidence": result.confidence,
                    "processing_time": result.processing_time
                }
            })
            successful += 1
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total_requests": len(requests),
        "successful": successful,
        "results": results
    }

if __name__ == "__main__":
    logger.info("🚀 KLP-SignGlove API 서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
