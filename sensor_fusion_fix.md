# 🔧 센서 융합 필터 수정 가이드

## 🚨 **현재 문제점**

### **잘못된 이중 융합 구조:**
```
Arduino INO: Raw 자이로/가속도 → 상보필터 → 각도 데이터
Python: 각도 데이터 → Madgwick 필터 → 최종 각도
```

**문제:**
- 이미 상보필터로 융합된 각도 데이터를 Madgwick에 입력
- Madgwick은 원시 자이로/가속도 데이터를 기대하는데 각도 데이터를 받음
- **이중 융합**으로 인한 부정확한 각도 출력
- 센서 값 자체가 달라짐

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
// 원시 센서 데이터만 전송
void loop() {
    // 센서 읽기
    float accel_x = readAccelerometerX();
    float accel_y = readAccelerometerY();
    float accel_z = readAccelerometerZ();
    float gyro_x = readGyroscopeX();
    float gyro_y = readGyroscopeY();
    float gyro_z = readGyroscopeZ();
    
    // 원시 데이터 전송 (융합 없음)
    Serial.print("ACCEL:"); Serial.print(accel_x); Serial.print(",");
    Serial.print(accel_y); Serial.print(","); Serial.print(accel_z);
    Serial.print(",GYRO:"); Serial.print(gyro_x); Serial.print(",");
    Serial.print(gyro_y); Serial.print(","); Serial.println(gyro_z);
}
```

#### Python 코드:
```python
# Madgwick 필터로 융합
from madgwick_filter import MadgwickFilter

madgwick = MadgwickFilter()

def process_sensor_data(data):
    # 원시 센서 데이터
    accel = [data['accel_x'], data['accel_y'], data['accel_z']]
    gyro = [data['gyro_x'], data['gyro_y'], data['gyro_z']]
    
    # Madgwick 융합
    orientation = madgwick.update(accel, gyro, dt)
    return orientation
```

### **방안 3: 선택적 사용**

#### 설정 기반 선택:
```python
class SensorFusionConfig:
    def __init__(self, fusion_method='complementary'):
        self.fusion_method = fusion_method
        
    def process_data(self, data):
        if self.fusion_method == 'complementary':
            # 상보필터 데이터 사용
            return self._use_complementary_data(data)
        elif self.fusion_method == 'madgwick':
            # Madgwick 융합
            return self._use_madgwick_fusion(data)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
```

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

## 📊 **성능 비교**

### **상보필터 (권장)**
- ✅ **빠른 처리**: 실시간 처리 가능
- ✅ **안정적**: 노이즈에 강함
- ✅ **간단함**: 구현 및 디버깅 용이
- ✅ **호환성**: 기존 모델과 호환

### **Madgwick 필터**
- ✅ **정확도**: 높은 정확도
- ❌ **복잡성**: 구현 복잡
- ❌ **계산량**: 더 많은 계산 필요
- ❌ **호환성**: 기존 모델 재학습 필요

## 🎯 **권장사항**

1. **즉시 상보필터만 사용**하도록 변경
2. **Madgwick 관련 코드 제거** 또는 비활성화
3. **기존 모델 재검증** 필요
4. **새로운 데이터 수집** 고려

## 🔄 **수정 단계**

1. **Arduino INO 코드 수정** (상보필터만 사용)
2. **Python 추론 코드 수정** (이중 융합 제거)
3. **테스트 및 검증**
4. **성능 측정 및 비교**
5. **필요시 모델 재학습**

---

*이 문서는 센서 융합 필터의 올바른 구현을 위한 가이드입니다.*
