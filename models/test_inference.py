#!/usr/bin/env python3
"""
Simple Inference Test for EGRU Model
"""

import torch
import numpy as np
import h5py
import os
from benchmark_300_epochs_model import BenchmarkEnhancedGRU

def test_inference():
    """Test inference with trained model"""
    print("🧪 Testing EGRU Model Inference")
    print("=" * 50)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Create model
    model = BenchmarkEnhancedGRU(
        input_size=24,
        hidden_size=32,
        num_layers=1,
        num_classes=24
    )
    
    # Load trained weights
    checkpoint = torch.load('benchmark_enhanced_gru_300epochs.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Test with sample data
    print("\n🔍 Testing with sample data...")
    
    # Create dummy test data (24 features, 300 time steps)
    test_data = np.random.randn(1, 300, 24).astype(np.float32)
    test_tensor = torch.FloatTensor(test_data).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Korean jamo mapping
    korean_classes = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ',
                     'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                     'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ',
                     'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    
    print(f"🎯 Predicted Class: {korean_classes[predicted_class]} ({predicted_class})")
    print(f"🎲 Confidence: {confidence:.1%}")
    
    # Test with real data if available
    print("\n🔍 Testing with real data...")
    data_path = "../data/unified"
    
    if os.path.exists(data_path):
        # Find first H5 file
        h5_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.h5'):
                    h5_files.append(os.path.join(root, file))
                    break
            if h5_files:
                break
        
        if h5_files:
            test_file = h5_files[0]
            print(f"📁 Testing with: {os.path.basename(test_file)}")
            
            try:
                with h5py.File(test_file, 'r') as f:
                    sensor_data = f['sensor_data'][:]
                    
                    if sensor_data.shape == (300, 8):
                        # Extract motion features
                        original = sensor_data
                        velocity = np.diff(sensor_data, axis=0, prepend=sensor_data[0:1])
                        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
                        features = np.concatenate([original, velocity, acceleration], axis=1)
                        
                        # Inference
                        test_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model(test_tensor)
                            probabilities = torch.softmax(outputs, dim=1)
                            predicted_class = torch.argmax(outputs, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                        
                        # Extract true label from file path
                        parts = test_file.split('/')
                        try:
                            class_idx = parts.index('unified') + 1
                            if class_idx < len(parts):
                                true_class = parts[class_idx]
                                print(f"🇰🇷 True Class: {true_class}")
                                print(f"🎯 Predicted: {korean_classes[predicted_class]}")
                                print(f"🎲 Confidence: {confidence:.1%}")
                                print(f"✅ Correct: {true_class == korean_classes[predicted_class]}")
                        except:
                            print(f"🎯 Predicted: {korean_classes[predicted_class]}")
                            print(f"🎲 Confidence: {confidence:.1%}")
                            
            except Exception as e:
                print(f"❌ Error testing with real data: {e}")
        else:
            print("❌ No H5 files found for testing")
    else:
        print("❌ Data path not found")
    
    print("\n🎉 Inference test completed!")

if __name__ == "__main__":
    test_inference()
