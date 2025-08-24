"""
SignGlove 추론 API 서버
FastAPI를 사용한 실시간 수화 인식 API
"""

import os
import sys
import time
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

# FastAPI 및 관련 라이브러리
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.unified_inference import (
    UnifiedInferencePipeline, 
    SensorReading, 
    InferenceMode,
    create_unified_inference_pipeline
)
from training.label_mapping import KSLLabelMapper
from word_recognition import WordRecognitionSystem

# Pydantic 모델 정의
class SensorDataRequest(BaseModel):
    """센서 데이터 요청 모델"""
    timestamp: float = Field(..., description="타임스탬프 (초)")
    pitch: float = Field(..., description="피치 각도 (도)")
    roll: float = Field(..., description="롤 각도 (도)")
    yaw: float = Field(..., description="요 각도 (도)")
    flex1: int = Field(..., description="플렉스 센서 1 값")
    flex2: int = Field(..., description="플렉스 센서 2 값")
    flex3: int = Field(..., description="플렉스 센서 3 값")
    flex4: int = Field(..., description="플렉스 센서 4 값")
    flex5: int = Field(..., description="플렉스 센서 5 값")
    source: str = Field(default="api", description="데이터 소스")

class BatchSensorDataRequest(BaseModel):
    """배치 센서 데이터 요청 모델"""
    sensor_data: List[SensorDataRequest] = Field(..., description="센서 데이터 배열")
    window_size: int = Field(default=20, description="윈도우 크기")
    stride: int = Field(default=10, description="스트라이드")

class PredictionResponse(BaseModel):
    """예측 응답 모델"""
    predicted_class: str = Field(..., description="예측된 클래스")
    confidence: float = Field(..., description="예측 신뢰도 (0.0-1.0)")
    stability_score: float = Field(..., description="안정성 점수")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")
    timestamp: float = Field(..., description="응답 타임스탬프")

class ModelInfoResponse(BaseModel):
    """모델 정보 응답 모델"""
    model_name: str = Field(..., description="모델 이름")
    model_version: str = Field(..., description="모델 버전")
    accuracy: float = Field(..., description="테스트 정확도")
    num_classes: int = Field(..., description="지원 클래스 수")
    supported_classes: List[str] = Field(..., description="지원 클래스 목록")
    input_features: int = Field(..., description="입력 특성 수")
    window_size: int = Field(..., description="윈도우 크기")

class PerformanceStatsResponse(BaseModel):
    """성능 통계 응답 모델"""
    fps: float = Field(..., description="초당 프레임 수")
    avg_latency_ms: float = Field(..., description="평균 지연시간 (밀리초)")
    total_predictions: int = Field(..., description="총 예측 수")
    buffer_utilization: float = Field(..., description="버퍼 사용률")
    confidence_threshold: float = Field(..., description="신뢰도 임계값")

# FastAPI 앱 생성
app = FastAPI(
    title="SignGlove 추론 API",
    description="실시간 한국수어 인식을 위한 추론 API 서버",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
inference_pipeline: Optional[UnifiedInferencePipeline] = None
label_mapper: Optional[KSLLabelMapper] = None
model_info: Dict[str, Any] = {}
word_recognition_system: Optional[WordRecognitionSystem] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global inference_pipeline, label_mapper, model_info, word_recognition_system
    
    print("🚀 SignGlove 추론 API 서버 시작 중...")
    
    try:
        # 라벨 매퍼 초기화
        label_mapper = KSLLabelMapper()
        
        # 추론 파이프라인 초기화 (새로운 Episode 모델 사용)
        model_path = "best_balanced_episode_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        inference_pipeline = create_unified_inference_pipeline(
            model_path=model_path,
            config_path=None
        )
        
        # 모델 정보 설정 (새로운 Episode 모델 정보)
        model_info = {
            "model_name": "SignGlove Balanced Episode Model",
            "model_version": "2.0.0",
            "accuracy": 0.9989,  # 새로운 테스트 정확도
            "num_classes": 24,
            "supported_classes": list(label_mapper.class_to_id.keys()),
            "input_features": 8,
            "window_size": 20
        }
        
        # 단어 인식 시스템 초기화
        word_recognition_system = WordRecognitionSystem()
        
        print("✅ API 서버 초기화 완료")
        print(f"📊 모델 정보: {model_info['model_name']} v{model_info['model_version']}")
        print(f"🎯 정확도: {model_info['accuracy']:.2%}")
        print(f"📈 지원 클래스: {model_info['num_classes']}개")
        print(f"📝 단어 인식 시스템: 활성화")
        
    except Exception as e:
        print(f"❌ 서버 초기화 실패: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    global inference_pipeline
    
    if inference_pipeline:
        inference_pipeline.stop_realtime_inference()
        print("🛑 추론 파이프라인 종료")

@app.get("/", response_model=Dict[str, str])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "SignGlove 추론 API 서버",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """헬스 체크 엔드포인트"""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    # 성능 통계 가져오기
    stats = inference_pipeline.get_performance_stats()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": True,
        "performance_stats": stats
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """모델 정보 조회"""
    global model_info
    
    if not model_info:
        raise HTTPException(status_code=503, detail="모델 정보를 가져올 수 없습니다")
    
    return ModelInfoResponse(**model_info)

@app.get("/model/performance", response_model=PerformanceStatsResponse)
async def get_performance_stats():
    """성능 통계 조회"""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    stats = inference_pipeline.get_performance_stats()
    return PerformanceStatsResponse(**stats)

@app.post("/predict", response_model=PredictionResponse)
async def predict_gesture(request: SensorDataRequest):
    """
    단일 센서 데이터로 제스처 예측
    
    Args:
        request: 센서 데이터 요청
        
    Returns:
        PredictionResponse: 예측 결과
    """
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    start_time = time.time()
    
    try:
        # 센서 데이터를 SensorReading으로 변환
        sensor_reading = SensorReading(
            timestamp=request.timestamp,
            flex_data=[request.flex1, request.flex2, request.flex3, request.flex4, request.flex5],
            orientation_data=[request.pitch, request.roll, request.yaw],
            source=request.source
        )
        
        # 센서 데이터 추가
        success = inference_pipeline.add_sensor_data(sensor_reading)
        
        if not success:
            raise HTTPException(status_code=400, detail="센서 데이터 추가 실패")
        
        # 예측 수행
        result = inference_pipeline.predict_single()
        
        if result is None:
            raise HTTPException(status_code=400, detail="예측을 수행할 수 없습니다. 충분한 데이터가 필요합니다.")
        
        processing_time = (time.time() - start_time) * 1000  # 밀리초로 변환
        
        return PredictionResponse(
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            stability_score=result.stability_score,
            processing_time_ms=processing_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_gesture_batch(request: BatchSensorDataRequest):
    """
    배치 센서 데이터로 제스처 예측
    
    Args:
        request: 배치 센서 데이터 요청
        
    Returns:
        List[PredictionResponse]: 예측 결과 배열
    """
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    if len(request.sensor_data) == 0:
        raise HTTPException(status_code=400, detail="센서 데이터가 비어있습니다")
    
    results = []
    
    try:
        for sensor_request in request.sensor_data:
            start_time = time.time()
            
            # 센서 데이터를 SensorReading으로 변환
            sensor_reading = SensorReading(
                timestamp=sensor_request.timestamp,
                flex_data=[sensor_request.flex1, sensor_request.flex2, sensor_request.flex3, sensor_request.flex4, sensor_request.flex5],
                orientation_data=[sensor_request.pitch, sensor_request.roll, sensor_request.yaw],
                source=sensor_request.source
            )
            
            # 센서 데이터 추가
            success = inference_pipeline.add_sensor_data(sensor_reading)
            
            if not success:
                continue
            
            # 예측 수행
            result = inference_pipeline.predict_single()
            
            if result is not None:
                processing_time = (time.time() - start_time) * 1000
                
                results.append(PredictionResponse(
                    predicted_class=result.predicted_class,
                    confidence=result.confidence,
                    stability_score=result.stability_score,
                    processing_time_ms=processing_time,
                    timestamp=time.time()
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 예측 중 오류 발생: {str(e)}")

@app.post("/predict/stable", response_model=PredictionResponse)
async def predict_stable_gesture(request: SensorDataRequest):
    """
    안정적인 제스처 예측 (안정성 체크 포함)
    
    Args:
        request: 센서 데이터 요청
        
    Returns:
        PredictionResponse: 안정적인 예측 결과
    """
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    start_time = time.time()
    
    try:
        # 센서 데이터를 SensorReading으로 변환
        sensor_reading = SensorReading(
            timestamp=request.timestamp,
            flex_data=[request.flex1, request.flex2, request.flex3, request.flex4, request.flex5],
            orientation_data=[request.pitch, request.roll, request.yaw],
            source=request.source
        )
        
        # 센서 데이터 추가
        success = inference_pipeline.add_sensor_data(sensor_reading)
        
        if not success:
            raise HTTPException(status_code=400, detail="센서 데이터 추가 실패")
        
        # 안정적인 예측 수행
        result = inference_pipeline.get_stable_prediction()
        
        if result is None:
            raise HTTPException(status_code=400, detail="안정적인 예측을 수행할 수 없습니다. 더 많은 데이터가 필요합니다.")
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            stability_score=result.stability_score,
            processing_time_ms=processing_time,
            timestamp=time.time()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"안정적 예측 중 오류 발생: {str(e)}")

@app.post("/config/confidence")
async def set_confidence_threshold(request: Dict[str, float]):
    threshold = request.get("threshold", 0.7)
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="임계값은 0.0과 1.0 사이여야 합니다")
    """
    신뢰도 임계값 설정
    
    Args:
        threshold: 신뢰도 임계값 (0.0-1.0)
    """
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    try:
        inference_pipeline.set_confidence_threshold(threshold)
        return {"message": f"신뢰도 임계값이 {threshold}로 설정되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"임계값 설정 중 오류 발생: {str(e)}")

@app.post("/buffer/clear")
async def clear_buffers():
    """버퍼 초기화"""
    global inference_pipeline
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    try:
        inference_pipeline.clear_buffers()
        return {"message": "버퍼가 초기화되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"버퍼 초기화 중 오류 발생: {str(e)}")

@app.get("/classes", response_model=Dict[str, List[str]])
async def get_supported_classes():
    """지원 클래스 목록 조회"""
    global label_mapper
    
    if label_mapper is None:
        raise HTTPException(status_code=503, detail="라벨 매퍼가 초기화되지 않았습니다")
    
    return {
        "consonants": label_mapper.get_consonants(),
        "vowels": label_mapper.get_vowels(),
        "all_classes": list(label_mapper.class_to_id.keys())
    }

@app.post("/word/recognize")
async def recognize_word(request: SensorDataRequest):
    """단어 인식"""
    global word_recognition_system, inference_pipeline
    
    if word_recognition_system is None:
        raise HTTPException(status_code=503, detail="단어 인식 시스템이 초기화되지 않았습니다")
    
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="추론 파이프라인이 초기화되지 않았습니다")
    
    try:
        start_time = time.time()
        
        # 센서 데이터를 SensorReading으로 변환
        sensor_reading = SensorReading(
            timestamp=request.timestamp,
            flex_data=[request.flex1, request.flex2, request.flex3, request.flex4, request.flex5],
            orientation_data=[request.pitch, request.roll, request.yaw],
            source=request.source
        )
        
        # 센서 데이터 추가
        success = inference_pipeline.add_sensor_data(sensor_reading)
        
        if not success:
            raise HTTPException(status_code=400, detail="센서 데이터 추가 실패")
        
        # 예측 수행
        result = inference_pipeline.predict_single()
        
        if result is None:
            # 충분한 데이터가 없으면 현재 상태만 반환
            status = word_recognition_system.get_current_status()
            return {
                "success": True,
                "letter_result": None,
                "word_result": None,
                "current_status": status,
                "timestamp": time.time()
            }
        
        # 글자 인식 결과
        letter = result.predicted_class
        confidence = result.confidence
        
        # 단어 인식 시스템에 글자 추가
        word_result = word_recognition_system.add_letter(letter, confidence, time.time())
        
        # 현재 상태
        status = word_recognition_system.get_current_status()
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "letter_result": {
                "letter": letter,
                "confidence": confidence,
                "timestamp": time.time()
            },
            "word_result": {
                "word": word_result.word if word_result else None,
                "confidence": word_result.confidence if word_result else None,
                "timestamp": word_result.timestamp if word_result else None
            },
            "current_status": status,
            "processing_time_ms": processing_time,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"단어 인식 중 오류 발생: {str(e)}")

@app.get("/word/status")
async def get_word_status():
    """단어 인식 상태 조회"""
    global word_recognition_system
    
    if word_recognition_system is None:
        raise HTTPException(status_code=503, detail="단어 인식 시스템이 초기화되지 않았습니다")
    
    status = word_recognition_system.get_current_status()
    
    return {
        "status": status,
        "timestamp": time.time()
    }

@app.post("/word/clear")
async def clear_current_word():
    """현재 단어 초기화"""
    global word_recognition_system
    
    if word_recognition_system is None:
        raise HTTPException(status_code=503, detail="단어 인식 시스템이 초기화되지 않았습니다")
    
    word_recognition_system.clear_current_word()
    
    return {
        "message": "현재 단어가 초기화되었습니다",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # 서버 실행
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
