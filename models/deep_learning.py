import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DeepLearningPipeline(nn.Module):
    """
    CNN + LSTM 기반의 한국수어 분류 모델
    입력: (batch_size, sequence_length, features) - 시계열 센서 데이터
    출력: 분류 결과 및 특징 벡터
    """
    def __init__(self, input_features=8, sequence_length=20, num_classes=5, 
                 hidden_dim=128, num_layers=2, dropout=0.3):
        super(DeepLearningPipeline, self).__init__()
        
        self.input_features = input_features  # flex1-5, pitch, roll, yaw
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 1D CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(256, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        # Feature extraction head
        self.feature_extractor = nn.Linear(hidden_dim, 64)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_features)
        Returns:
            dict with 'class_logits' and 'features'
        """
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        # Transpose back for LSTM: (batch, sequence, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        class_logits = self.classifier(attended_output)
        
        # Feature extraction
        features = self.feature_extractor(attended_output)
        
        return {
            'class_logits': class_logits,
            'features': features,
            'attention_weights': attention_weights
        }
    
    def get_feature_importance(self, x):
        """특징 중요도 분석을 위한 메서드"""
        with torch.no_grad():
            output = self.forward(x)
            attention_weights = output['attention_weights']
            return attention_weights.cpu().numpy()

class CNNLSTMAdvanced(nn.Module):
    """
    더 고급 CNN-LSTM 모델 (옵션)
    """
    def __init__(self, input_features=8, sequence_length=20, num_classes=5):
        super(CNNLSTMAdvanced, self).__init__()
        
        # Multi-scale CNN
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_features, 64, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ) for k in [3, 5, 7]
        ])
        
        # Feature fusion
        self.fusion = nn.Conv1d(64*3, 256, kernel_size=1)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(256, 128, 2, batch_first=True, 
                           dropout=0.3, bidirectional=True)
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(256, 8, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Multi-scale feature extraction
        x = x.transpose(1, 2)  # (batch, features, sequence)
        
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Concatenate multi-scale features
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)
        
        # Transpose for LSTM
        x = x.transpose(1, 2)  # (batch, sequence, features)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        class_logits = self.classifier(pooled)
        
        return {'class_logits': class_logits}
