# 🔧 센서 융합 필터 수정 가이드

*이 문서는 센서 융합 필터의 올바른 구현을 위한 가이드입니다.*

## 🚨 **현재 문제점**

### **1. 잘못된 이중 융합 구조:**
```
Arduino INO: Raw 자이로/가속도 → 상보필터 → 각도 데이터
Python: 각도 데이터 → Madgwick 필터 → 최종 각도
```

**문제:**
- 이미 상보필터로 융합된 각도 데이터를 Madgwick에 입력
- Madgwick은 원시 자이로/가속도 데이터를 기대하는데 각도 데이터를 받음
- **이중 융합**으로 인한 부정확한 각도 출력
- 센서 값 자체가 달라짐

### **2. 학습 데이터 불일치 문제:**
```
기존 학습 데이터 (ㄱ_sample_data.csv):
- 각도 범위: -1° ~ +1° (작은 각도)
- 예시: pitch: -0.85°, roll: -0.06°, yaw: 0.12°

새로운 추론 데이터 (ㄱ_unified_data_066.csv):
- 각도 범위: -45° ~ +45° (큰 각도)
- 예시: pitch: 15.55°, roll: -44.59°, yaw: 8.82°

결과: 모델이 큰 각도 데이터를 인식하지 못함
```

## ✅ **올바른 해결 방안**

### **방안 1: 상보필터만 사용 (권장)**

#### Arduino INO 코드:
```cpp
// 상보필터만 사용
float complementaryFilter(float accel_angle, float gyro_rate, float dt) {
    static float angle = 0;
    float alpha = 0.96;  // 상보필터 계수
    
    angle = alpha * (angle + gyro_rate * dt) + (1 - alpha) * accel_angle;
    return angle;
}

void loop() {
    // 센서 읽기
    float accel_x = readAccelerometerX();
    float accel_y = readAccelerometerY();
    float gyro_x = readGyroscopeX();
    float gyro_y = readGyroscopeY();
    
    // 각도 계산
    float accel_pitch = atan2(accel_y, accel_x) * RAD_TO_DEG;
    float accel_roll = atan2(accel_x, accel_z) * RAD_TO_DEG;
    
    // 상보필터 적용
    float pitch = complementaryFilter(accel_pitch, gyro_x, dt);
    float roll = complementaryFilter(accel_roll, gyro_y, dt);
    
    // Python으로 전송 (각도 데이터만)
    Serial.print("PITCH:"); Serial.print(pitch);
    Serial.print(",ROLL:"); Serial.print(roll);
    Serial.print(",YAW:"); Serial.println(yaw);
}
```

#### Python 코드:
```python
# 상보필터 데이터만 받기
def process_sensor_data(data):
    # 이미 융합된 각도 데이터 사용
    pitch = data['pitch']
    roll = data['roll'] 
    yaw = data['yaw']
    
    # 추가 융합 없이 바로 사용
    orientation_data = [pitch, roll, yaw]
    return orientation_data
```

### **방안 2: Madgwick만 사용**

#### Arduino INO 코드:
```cpp
// 원시 데이터만 전송
void loop() {
    // 센서 읽기
    float ax = readAccelerometerX();
    float ay = readAccelerometerY();
    float az = readAccelerometerZ();
    float gx = readGyroscopeX();
    float gy = readGyroscopeY();
    float gz = readGyroscopeZ();
    
    // 원시 데이터 전송 (Madgwick용)
    Serial.print("AX:"); Serial.print(ax);
    Serial.print(",AY:"); Serial.print(ay);
    Serial.print(",AZ:"); Serial.print(az);
    Serial.print(",GX:"); Serial.print(gx);
    Serial.print(",GY:"); Serial.print(gy);
    Serial.print(",GZ:"); Serial.println(gz);
}
```

#### Python 코드:
```python
# Madgwick 필터 사용
from madgwick import MadgwickFilter

madgwick = MadgwickFilter()

def process_sensor_data(data):
    # Madgwick 융합 수행
    orientation = madgwick.update(
        [data['ax'], data['ay'], data['az']],  # 가속도
        [data['gx'], data['gy'], data['gz']],  # 자이로
        data['dt']  # 시간 간격
    )
    return orientation
```

### **방안 3: 선택적 사용**

#### Python 설정:
```python
class UnifiedInferencePipeline:
    def __init__(self, fusion_method='complementary'):
        self.fusion_method = fusion_method
        if fusion_method == 'madgwick':
            self.madgwick_filter = MadgwickFilter()
    
    def process_sensor_data(self, data):
        if self.fusion_method == 'complementary':
            # 이미 융합된 각도 데이터 사용
            return data['orientation']
        elif self.fusion_method == 'madgwick':
            # Madgwick 융합 수행
            return self.madgwick_filter.update(
                data['accel'], data['gyro'], data['dt']
            )
```

#### 설정 파일:
```json
{
  "sensor_fusion": {
    "method": "complementary",  // "complementary" 또는 "madgwick"
    "complementary_alpha": 0.96,
    "madgwick_beta": 0.1
  }
}
```

## 🎯 **학습 데이터 재구성 필요**

### **문제:**
- 기존 모델은 작은 각도 데이터로 학습됨
- 새로운 상보필터 데이터는 큰 각도 범위
- **모델 재학습 필요**

### **해결책:**
```bash
# 상보필터 전용 모델 학습
python training/train_complementary_filter.py
```

### **새로운 학습 스크립트 특징:**
- **Unified 데이터셋 사용**: `*_unified_data_*.csv` 파일들
- **상보필터 데이터 최적화**: 큰 각도 범위에 맞춤
- **24개 클래스 지원**: 14개 자음 + 10개 모음
- **자동 데이터 분할**: 훈련/검증/테스트
- **Early stopping**: 과적합 방지
- **상세 평가**: 분류 보고서, 혼동 행렬

## 📊 **성능 비교**

| 필터 방식 | 장점 | 단점 | 추천도 |
|-----------|------|------|--------|
| **상보필터** | ✅ 간단하고 빠름<br>✅ 안정적<br>✅ 메모리 효율적 | ❌ 정확도 제한적 | ⭐⭐⭐⭐⭐ |
| **Madgwick** | ✅ 높은 정확도<br>✅ 노이즈에 강함 | ❌ 복잡함<br>❌ 계산량 많음 | ⭐⭐⭐ |

## 🔧 **즉시 수정해야 할 파일들**

### 1. **Arduino INO 코드 수정**
- 상보필터만 사용하도록 변경
- 원시 데이터 전송 옵션 추가

### 2. **Python 추론 코드 수정**
```python
# unified_inference.py 수정
class UnifiedInferencePipeline:
    def __init__(self, fusion_method='complementary'):
        self.fusion_method = fusion_method
        if fusion_method == 'madgwick':
            self.madgwick_filter = MadgwickFilter()
    
    def process_sensor_data(self, data):
        if self.fusion_method == 'complementary':
            # 이미 융합된 각도 데이터 사용
            return data['orientation']
        elif self.fusion_method == 'madgwick':
            # Madgwick 융합 수행
            return self.madgwick_filter.update(
                data['accel'], data['gyro'], data['dt']
            )
```

### 3. **설정 파일 수정**
```json
{
  "sensor_fusion": {
    "method": "complementary",  // "complementary" 또는 "madgwick"
    "complementary_alpha": 0.96,
    "madgwick_beta": 0.1
  }
}
```

## 🔄 **수정 단계**

1. **Arduino INO 코드 수정** (상보필터만 사용)
2. **Python 추론 코드 수정** (이중 융합 제거)
3. **새로운 모델 학습** (상보필터 데이터로)
4. **테스트 및 검증**
5. **성능 측정 및 비교**

## 🎯 **권장사항**

### **즉시 실행:**
1. ✅ **상보필터만 사용** (이중 융합 제거)
2. ✅ **Madgwick 코드 제거/비활성화**
3. ✅ **기존 모델 재검증**

### **단기 계획:**
1. 🔄 **새로운 데이터로 모델 재학습**
2. 🔄 **상보필터 전용 모델 생성**
3. 🔄 **성능 비교 및 최적화**

### **장기 계획:**
1. 📈 **데이터 수집 확대** (더 많은 각도 범위)
2. 📈 **모델 아키텍처 개선**
3. 📈 **실시간 성능 최적화**

## 📝 **결론**

**상보필터만 사용하는 것이 가장 좋은 선택**입니다:

- ✅ **단순하고 안정적**
- ✅ **이중 융합 문제 해결**
- ✅ **기존 하드웨어와 호환**
- ✅ **새로운 모델 학습으로 정확도 향상 가능**

**즉시 상보필터 전용으로 전환하고, 새로운 데이터로 모델을 재학습하는 것을 권장합니다.**
