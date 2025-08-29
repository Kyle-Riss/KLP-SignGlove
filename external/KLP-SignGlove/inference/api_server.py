from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import json
import time
import logging
from typing import List, Dict, Optional
import numpy as np
import asyncio
from pathlib import Path

# 추론 시스템 import
from signglove_inference import SignGloveInference

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="SignGlove API",
    description="실시간 한국수어 인식 API 서버",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
inference_system = None
config = None

# Pydantic 모델
class SensorData(BaseModel):
    """센서 데이터 모델"""
    flex1: float
    flex2: float
    flex3: float
    flex4: float
    flex5: float
    pitch: float
    roll: float
    yaw: float

class PredictionRequest(BaseModel):
    """예측 요청 모델"""
    sensor_data: List[SensorData]

class PredictionResponse(BaseModel):
    """예측 응답 모델"""
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: float

class SystemInfo(BaseModel):
    """시스템 정보 모델"""
    model_path: str
    device: str
    total_parameters: int
    labels: List[str]
    config: Dict

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global inference_system, config
    
    try:
        # 설정 파일 로드
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 모델 경로 설정
        model_path = Path(__file__).parent.parent / "cross_validation_model.pth"
        
        # 추론 시스템 초기화
        inference_system = SignGloveInference(
            model_path=str(model_path),
            config_path=str(config_path)
        )
        
        logger.info("SignGlove API 서버 초기화 완료")
        
    except Exception as e:
        logger.error(f"서버 초기화 실패: {e}")
        raise

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "SignGlove API 서버",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": inference_system is not None
    }

@app.get("/info", response_model=SystemInfo)
async def get_system_info():
    """시스템 정보 조회"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="추론 시스템이 초기화되지 않았습니다")
    
    model_info = inference_system.get_model_info()
    
    return SystemInfo(
        model_path=model_info['model_path'],
        device=model_info['device'],
        total_parameters=model_info['total_parameters'],
        labels=list(model_info['label_mapper'].values()),
        config=model_info['config']
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """단일 예측"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="추론 시스템이 초기화되지 않았습니다")
    
    try:
        # 센서 데이터를 numpy 배열로 변환
        sensor_data = []
        for data in request.sensor_data:
            sensor_data.append([
                data.flex1, data.flex2, data.flex3, data.flex4, data.flex5,
                data.pitch, data.roll, data.yaw
            ])
        
        sensor_array = np.array(sensor_data)
        
        # 예측 수행
        result = inference_system.predict_single(sensor_array)
        
        # 확률을 라벨과 매핑
        probabilities = {}
        for i, prob in enumerate(result['probabilities']):
            label = inference_system.label_mapper[i]
            probabilities[label] = float(prob)
        
        return PredictionResponse(
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            probabilities=probabilities,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"예측 오류: {e}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(request: PredictionRequest):
    """배치 예측"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="추론 시스템이 초기화되지 않았습니다")
    
    try:
        # 센서 데이터를 시퀀스로 그룹화 (20개씩)
        sequence_length = inference_system.config['sequence_length']
        sensor_sequences = []
        
        for i in range(0, len(request.sensor_data), sequence_length):
            sequence_data = request.sensor_data[i:i+sequence_length]
            if len(sequence_data) == sequence_length:
                # 시퀀스를 numpy 배열로 변환
                sensor_sequence = []
                for data in sequence_data:
                    sensor_sequence.append([
                        data.flex1, data.flex2, data.flex3, data.flex4, data.flex5,
                        data.pitch, data.roll, data.yaw
                    ])
                sensor_sequences.append(np.array(sensor_sequence))
        
        # 각 시퀀스에 대해 예측 수행
        results = []
        for sensor_sequence in sensor_sequences:
            result = inference_system.predict_single(sensor_sequence)
            results.append(result)
        
        # 결과 변환
        response_results = []
        for result in results:
            probabilities = {}
            for i, prob in enumerate(result['probabilities']):
                label = inference_system.label_mapper[i]
                probabilities[label] = float(prob)
            
            response_results.append({
                "predicted_label": result['predicted_label'],
                "confidence": result['confidence'],
                "probabilities": probabilities,
                "timestamp": time.time()
            })
        
        return {
            "results": response_results,
            "total_predictions": len(response_results)
        }
        
    except Exception as e:
        logger.error(f"배치 예측 오류: {e}")
        raise HTTPException(status_code=500, detail=f"배치 예측 중 오류 발생: {str(e)}")

@app.post("/realtime/start")
async def start_realtime_prediction():
    """실시간 예측 시작"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="추론 시스템이 초기화되지 않았습니다")
    
    # 버퍼 초기화
    inference_system.reset_buffer()
    
    return {
        "message": "실시간 예측 시작",
        "buffer_size": inference_system.buffer_size,
        "timestamp": time.time()
    }

@app.post("/realtime/add")
async def add_sensor_data(sensor_data: SensorData):
    """실시간 센서 데이터 추가"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="추론 시스템이 초기화되지 않았습니다")
    
    try:
        # 센서 데이터를 numpy 배열로 변환
        data_array = np.array([
            sensor_data.flex1, sensor_data.flex2, sensor_data.flex3, 
            sensor_data.flex4, sensor_data.flex5, sensor_data.pitch, 
            sensor_data.roll, sensor_data.yaw
        ])
        
        # 버퍼에 추가
        inference_system.add_sensor_data(data_array)
        
        # 실시간 예측 수행
        result = inference_system.predict_realtime()
        
        if result:
            # 확률을 라벨과 매핑
            probabilities = {}
            for i, prob in enumerate(result['probabilities']):
                label = inference_system.label_mapper[i]
                probabilities[label] = float(prob)
            
            return {
                "prediction": {
                    "predicted_label": result['predicted_label'],
                    "confidence": result['confidence'],
                    "probabilities": probabilities,
                    "timestamp": time.time()
                },
                "buffer_size": len(inference_system.sensor_buffer)
            }
        else:
            return {
                "prediction": None,
                "buffer_size": len(inference_system.sensor_buffer),
                "message": "버퍼가 아직 가득 차지 않았습니다"
            }
            
    except Exception as e:
        logger.error(f"실시간 예측 오류: {e}")
        raise HTTPException(status_code=500, detail=f"실시간 예측 중 오류 발생: {str(e)}")

@app.post("/realtime/reset")
async def reset_realtime_buffer():
    """실시간 버퍼 초기화"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="추론 시스템이 초기화되지 않았습니다")
    
    inference_system.reset_buffer()
    
    return {
        "message": "실시간 버퍼 초기화 완료",
        "timestamp": time.time()
    }

@app.get("/labels")
async def get_labels():
    """사용 가능한 라벨 목록 조회"""
    if inference_system is None:
        raise HTTPException(status_code=503, detail="추론 시스템이 초기화되지 않았습니다")
    
    return {
        "labels": list(inference_system.label_mapper.values()),
        "total_labels": len(inference_system.label_mapper)
    }

@app.get("/config")
async def get_config():
    """설정 정보 조회"""
    if config is None:
        raise HTTPException(status_code=503, detail="설정이 로드되지 않았습니다")
    
    return config

# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"전역 오류: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"서버 내부 오류: {str(exc)}"}
    )

def main():
    """메인 함수"""
    # 설정 로드
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    api_config = config['api_config']
    
    # 서버 실행
    uvicorn.run(
        "api_server:app",
        host=api_config['host'],
        port=api_config['port'],
        reload=api_config['debug'],
        workers=api_config['workers']
    )

if __name__ == "__main__":
    main()
