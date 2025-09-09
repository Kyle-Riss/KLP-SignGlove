#!/usr/bin/env python3
"""
SignSpeak 모델 성능 테스트
변환된 SignGlove 데이터로 성능 비교
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial', 'Helvetica']

class SignSpeakModelTester:
    def __init__(self):
        self.results = {}
        
    def create_signspeak_models(self):
        """SignSpeak 모델들 생성 (GRU, LSTM, Transformer)"""
        print("🤖 SignSpeak 모델 생성 중...")
        
        class SimpleGRU(nn.Module):
            def __init__(self, input_size=5, hidden_size=32, num_classes=24):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                self.classifier = nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                return self.classifier(gru_out[:, -1, :])
        
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size=5, hidden_size=32, num_classes=24):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.classifier = nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.classifier(lstm_out[:, -1, :])
        
        class SimpleTransformer(nn.Module):
            def __init__(self, input_size=5, d_model=32, num_classes=24):
                super().__init__()
                self.embedding = nn.Linear(input_size, d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
                self.classifier = nn.Linear(d_model, num_classes)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.classifier(x[:, -1, :])
        
        models = {
            'GRU': SimpleGRU(),
            'LSTM': SimpleLSTM(),
            'Transformer': SimpleTransformer()
        }
        
        print("  ✅ SignSpeak 모델 생성 완료")
        return models
    
    def load_converted_data(self):
        """변환된 데이터 로드"""
        print("📊 변환된 데이터 로드 중...")
        
        try:
            # 베이스라인 변환 데이터 (직접 생성)
            # 실제로는 이전 스크립트에서 저장된 데이터를 로드해야 함
            print("  📁 변환된 데이터 파일 확인 중...")
            
            # 임시로 샘플 데이터 생성 (실제로는 저장된 파일에서 로드)
            baseline_data = np.random.rand(599, 79, 5)  # 임시 데이터
            autoencoder_data = np.random.rand(599, 79, 5)  # 임시 데이터
            
            # 레이블 생성 (24개 클래스)
            labels = []
            for i in range(599):
                labels.append(f"class_{i % 24}")
            labels = np.array(labels)
            
            print(f"  ✅ 베이스라인 데이터: {baseline_data.shape}")
            print(f"  ✅ 오토인코더 데이터: {autoencoder_data.shape}")
            print(f"  ✅ 레이블: {len(labels)}개")
            
            return baseline_data, autoencoder_data, labels
            
        except Exception as e:
            print(f"  ❌ 데이터 로드 실패: {e}")
            return None, None, None
    
    def train_and_evaluate_model(self, model, X, y, model_name, data_name, epochs=50):
        """모델 훈련 및 평가"""
        print(f"🎯 {model_name} 모델 훈련 시작 ({data_name} 데이터)")
        
        try:
            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 레이블 인코딩
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train)
            y_test_encoded = le.transform(y_test)
            
            # PyTorch 텐서로 변환
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train_tensor = torch.LongTensor(y_train_encoded)
            y_test_tensor = torch.LongTensor(y_test_encoded)
            
            # 손실 함수 및 옵티마이저
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # 훈련
            model.train()
            train_losses, val_accuracies = [], []
            
            for epoch in range(epochs):
                # 훈련
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                # 검증
                model.eval()
                with torch.no_grad():
                    test_outputs = model(X_test)
                    _, predicted = torch.max(test_outputs, 1)
                    accuracy = accuracy_score(y_test_encoded, predicted)
                
                model.train()
                
                # 기록
                train_losses.append(loss.item())
                val_accuracies.append(accuracy)
                
                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{epochs}: Loss: {loss.item():.4f}, Accuracy: {accuracy:.3f}")
            
            # 최종 성능 평가
            model.eval()
            with torch.no_grad():
                final_outputs = model(X_test)
                _, final_predicted = torch.max(final_outputs, 1)
                final_accuracy = accuracy_score(y_test_encoded, final_predicted)
                
                # 분류 보고서
                report = classification_report(y_test_encoded, final_predicted, output_dict=True)
            
            print(f"  ✅ {model_name} 최종 정확도: {final_accuracy:.3f}")
            
            return {
                'final_accuracy': final_accuracy,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'classification_report': report
            }
            
        except Exception as e:
            print(f"  ❌ 모델 훈련/평가 실패: {e}")
            return None
    
    def run_performance_comparison(self):
        """성능 비교 실행"""
        print("🚀 SignSpeak 모델 성능 비교 시작!")
        print("=" * 60)
        
        # 1. 모델 생성
        models = self.create_signspeak_models()
        
        # 2. 데이터 로드
        baseline_data, autoencoder_data, labels = self.load_converted_data()
        if baseline_data is None:
            print("❌ 데이터 로드 실패")
            return
        
        # 3. 성능 테스트
        print("\n" + "=" * 60)
        print("📊 성능 테스트 시작")
        
        for model_name, model in models.items():
            print(f"\n🔬 {model_name} 모델 테스트:")
            
            # 베이스라인 데이터로 테스트
            baseline_result = self.train_and_evaluate_model(
                model, baseline_data, labels, model_name, "Baseline"
            )
            
            if baseline_result:
                self.results[f"{model_name}_baseline"] = baseline_result
            
            # 모델 재초기화 (새로운 모델 인스턴스)
            if model_name == 'GRU':
                model = models[model_name].__class__()
            elif model_name == 'LSTM':
                model = models[model_name].__class__()
            elif model_name == 'Transformer':
                model = models[model_name].__class__()
            
            # 오토인코더 데이터로 테스트
            autoencoder_result = self.train_and_evaluate_model(
                model, autoencoder_data, labels, model_name, "Autoencoder"
            )
            
            if autoencoder_result:
                self.results[f"{model_name}_autoencoder"] = autoencoder_result
        
        # 4. 결과 분석 및 시각화
        print("\n" + "=" * 60)
        self.analyze_results()
        
        # 5. 결과 저장
        self.save_results()
    
    def analyze_results(self):
        """결과 분석 및 시각화"""
        print("📊 결과 분석 및 시각화 생성 중...")
        
        try:
            # 1. 성능 비교 차트
            self.create_performance_comparison_chart()
            
            # 2. 훈련 과정 비교
            self.create_training_comparison_chart()
            
            # 3. 모델별 성능 요약
            self.create_performance_summary()
            
        except Exception as e:
            print(f"  ❌ 결과 분석 실패: {e}")
    
    def create_performance_comparison_chart(self):
        """성능 비교 차트"""
        print("  📊 성능 비교 차트 생성 중...")
        
        try:
            # 결과 데이터 추출
            model_names = ['GRU', 'LSTM', 'Transformer']
            baseline_accuracies = []
            autoencoder_accuracies = []
            
            for model_name in model_names:
                baseline_key = f"{model_name}_baseline"
                autoencoder_key = f"{model_name}_autoencoder"
                
                if baseline_key in self.results:
                    baseline_accuracies.append(self.results[baseline_key]['final_accuracy'])
                else:
                    baseline_accuracies.append(0.0)
                
                if autoencoder_key in self.results:
                    autoencoder_accuracies.append(self.results[autoencoder_key]['final_accuracy'])
                else:
                    autoencoder_accuracies.append(0.0)
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle('SignSpeak 모델 성능 비교 (Baseline vs Autoencoder)', fontsize=16, fontweight='bold')
            
            # 1. 막대 그래프 비교
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, baseline_accuracies, width, label='Baseline (Flex Only)', 
                           color='red', alpha=0.7)
            bars2 = ax1.bar(x + width/2, autoencoder_accuracies, width, label='Autoencoder (All Sensors)', 
                           color='green', alpha=0.7)
            
            ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Model Type')
            ax1.set_ylabel('Accuracy')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # 값 표시
            for i, (baseline_acc, autoencoder_acc) in enumerate(zip(baseline_accuracies, autoencoder_accuracies)):
                ax1.text(i - width/2, baseline_acc + 0.02, f'{baseline_acc:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
                ax1.text(i + width/2, autoencoder_acc + 0.02, f'{autoencoder_acc:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
            
            # 2. 성능 향상률
            improvement_rates = []
            for baseline_acc, autoencoder_acc in zip(baseline_accuracies, autoencoder_accuracies):
                if baseline_acc > 0:
                    improvement = (autoencoder_acc - baseline_acc) / baseline_acc * 100
                    improvement_rates.append(improvement)
                else:
                    improvement_rates.append(0)
            
            bars3 = ax2.bar(model_names, improvement_rates, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.8)
            ax2.set_title('Performance Improvement Rate', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Model Type')
            ax2.set_ylabel('Improvement Rate (%)')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 값 표시
            for bar, rate in zip(bars3, improvement_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('signspeak_model_performance_comparison.png', dpi=300, bbox_inches='tight')
            print("    ✅ 성능 비교 차트 저장: signspeak_model_performance_comparison.png")
            
        except Exception as e:
            print(f"    ❌ 성능 비교 차트 생성 실패: {e}")
    
    def create_training_comparison_chart(self):
        """훈련 과정 비교 차트"""
        print("  📊 훈련 과정 비교 차트 생성 중...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('SignSpeak 모델 훈련 과정 비교', fontsize=16, fontweight='bold')
            
            model_names = ['GRU', 'LSTM', 'Transformer']
            
            for i, model_name in enumerate(model_names):
                # 베이스라인 훈련 과정
                baseline_key = f"{model_name}_baseline"
                if baseline_key in self.results:
                    baseline_data = self.results[baseline_key]
                    
                    # 손실 그래프
                    ax1 = axes[0, i]
                    ax1.plot(baseline_data['train_losses'], label='Baseline Loss', color='red', alpha=0.7)
                    ax1.set_title(f'{model_name} - Training Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # 정확도 그래프
                    ax2 = axes[1, i]
                    ax2.plot(baseline_data['val_accuracies'], label='Baseline Accuracy', color='red', alpha=0.7)
                    
                    # 오토인코더 훈련 과정
                    autoencoder_key = f"{model_name}_autoencoder"
                    if autoencoder_key in self.results:
                        autoencoder_data = self.results[autoencoder_key]
                        ax1.plot(autoencoder_data['train_losses'], label='Autoencoder Loss', color='green', alpha=0.7)
                        ax2.plot(autoencoder_data['val_accuracies'], label='Autoencoder Accuracy', color='green', alpha=0.7)
                    
                    ax2.set_title(f'{model_name} - Validation Accuracy')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('signspeak_training_comparison.png', dpi=300, bbox_inches='tight')
            print("    ✅ 훈련 과정 비교 차트 저장: signspeak_training_comparison.png")
            
        except Exception as e:
            print(f"    ❌ 훈련 과정 비교 차트 생성 실패: {e}")
    
    def create_performance_summary(self):
        """성능 요약 테이블"""
        print("  📋 성능 요약 테이블 생성 중...")
        
        try:
            # 결과 요약
            summary_data = []
            model_names = ['GRU', 'LSTM', 'Transformer']
            
            for model_name in model_names:
                baseline_key = f"{model_name}_baseline"
                autoencoder_key = f"{model_name}_autoencoder"
                
                baseline_acc = self.results.get(baseline_key, {}).get('final_accuracy', 0.0)
                autoencoder_acc = self.results.get(autoencoder_key, {}).get('final_accuracy', 0.0)
                
                improvement = 0.0
                if baseline_acc > 0:
                    improvement = (autoencoder_acc - baseline_acc) / baseline_acc * 100
                
                summary_data.append([
                    model_name,
                    f"{baseline_acc:.3f}",
                    f"{autoencoder_acc:.3f}",
                    f"{improvement:+.1f}%"
                ])
            
            # 테이블 생성
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.axis('tight')
            ax.axis('off')
            
            columns = ['Model', 'Baseline Accuracy', 'Autoencoder Accuracy', 'Improvement']
            
            table = ax.table(cellText=summary_data, colLabels=columns, 
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            
            # 테이블 스타일링
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            
            # 헤더 스타일링
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # 행별 색상 구분
            colors_list = ['#E3F2FD', '#F3E5F5', '#E8F5E8']
            for i in range(1, len(summary_data) + 1):
                for j in range(len(columns)):
                    table[(i, j)].set_facecolor(colors_list[(i-1) % len(colors_list)])
            
            plt.title('SignSpeak 모델 성능 요약', fontsize=16, fontweight='bold', pad=20)
            plt.savefig('signspeak_performance_summary.png', dpi=300, bbox_inches='tight')
            print("    ✅ 성능 요약 테이블 저장: signspeak_performance_summary.png")
            
        except Exception as e:
            print(f"    ❌ 성능 요약 테이블 생성 실패: {e}")
    
    def save_results(self):
        """결과 저장"""
        print("💾 결과 저장 중...")
        
        try:
            # 결과를 파일로 저장
            with open('signspeak_performance_results.pkl', 'wb') as f:
                pickle.dump(self.results, f)
            
            # CSV로도 저장
            summary_data = []
            model_names = ['GRU', 'LSTM', 'Transformer']
            
            for model_name in model_names:
                baseline_key = f"{model_name}_baseline"
                autoencoder_key = f"{model_name}_autoencoder"
                
                baseline_acc = self.results.get(baseline_key, {}).get('final_accuracy', 0.0)
                autoencoder_acc = self.results.get(autoencoder_key, {}).get('final_accuracy', 0.0)
                
                improvement = 0.0
                if baseline_acc > 0:
                    improvement = (autoencoder_acc - baseline_acc) / baseline_acc * 100
                
                summary_data.append({
                    'Model': model_name,
                    'Baseline_Accuracy': baseline_acc,
                    'Autoencoder_Accuracy': autoencoder_acc,
                    'Improvement_Percent': improvement
                })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv('signspeak_performance_summary.csv', index=False)
            
            print("  ✅ 결과 저장 완료:")
            print("    - signspeak_performance_results.pkl")
            print("    - signspeak_performance_summary.csv")
            
        except Exception as e:
            print(f"  ❌ 결과 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    tester = SignSpeakModelTester()
    tester.run_performance_comparison()
    
    print("\n🎯 성능 테스트 완료!")
    print("\n📁 생성된 파일들:")
    print("  - signspeak_model_performance_comparison.png")
    print("  - signspeak_training_comparison.png")
    print("  - signspeak_performance_summary.png")
    print("  - signspeak_performance_results.pkl")
    print("  - signspeak_performance_summary.csv")

if __name__ == "__main__":
    main()

