"""
Amygdala-Boosted GRU (A-GRU)
편도체에서 영감을 받은 중요도 기반 기억 강화 메커니즘

핵심 아이디어:
- A-Net: 현재 입력의 분류적 중요도 계산 (편도체 역할)
- 입력 증폭: 중요한 입력을 강화하여 GRU에 전달
- 강화된 기억: 중요한 정보가 더 잘 저장되도록 유도
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AmygdalaNet(nn.Module):
    """
    A-Net: 편도체 네트워크
    입력 X_t와 이전 은닉 상태 h_{t-1}을 보고 중요도 e_t 계산
    
    수식:
        e_t = σ(W_A [X_t ⊕ h_{t-1}] + b_A)
        X'_t = X_t ⊙ (1 + γ·e_t)
    """
    
    def __init__(self, input_size: int, hidden_size: int, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        
        # 입력과 은닉 상태를 연결해서 중요도 계산
        self.importance_net = nn.Sequential(
            nn.Linear(input_size + hidden_size, (input_size + hidden_size) // 2),
            nn.Tanh(),
            nn.Linear((input_size + hidden_size) // 2, input_size),
            nn.Sigmoid()  # 0~1 범위의 중요도
        )
    
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: (batch, input_size) - 현재 입력
            h_prev: (batch, hidden_size) - 이전 은닉 상태
        
        Returns:
            x_boosted: (batch, input_size) - 증폭된 입력
            importance: (batch, input_size) - 중요도 점수
        """
        # [X_t ⊕ h_{t-1}] 연결
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # e_t = σ(W_A [X_t ⊕ h_{t-1}] + b_A)
        importance = self.importance_net(combined)
        
        # X'_t = X_t ⊙ (1 + γ·e_t)
        x_boosted = x_t * (1.0 + self.gamma * importance)
        
        return x_boosted, importance


class AGRUCell(nn.Module):
    """
    A-GRU Cell: Amygdala-Boosted GRU Cell
    A-Net으로 증폭된 입력을 사용하는 GRU 셀
    
    수식:
        r_t = σ(W_Xr X'_t + W_Hr h_{t-1} + b_r)
        z_t = σ(W_Xz X'_t + W_Hz h_{t-1} + b_z)
        h̃_t = tanh(W_R (r_t ⊙ h_{t-1}) + W_X X'_t + b_h)
        h_t = z_t ⊙ h̃_t + (1 - z_t) ⊙ h_{t-1}
    """
    
    def __init__(self, input_size: int, hidden_size: int, gamma: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # A-Net: 편도체 네트워크
        self.a_net = AmygdalaNet(input_size, hidden_size, gamma)
        
        # GRU 게이트들 (표준 GRU와 동일하지만 X'_t 사용)
        # Reset gate
        self.W_xr = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Update gate
        self.W_xz = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Candidate hidden state
        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_r = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: (batch, input_size)
            h_prev: (batch, hidden_size)
        
        Returns:
            h_t: (batch, hidden_size) - 새로운 은닉 상태
            importance: (batch, input_size) - A-Net 중요도 (분석용)
        """
        # A-Net: X'_t = X_t ⊙ (1 + γ·e_t)
        x_boosted, importance = self.a_net(x_t, h_prev)
        
        # Reset gate: r_t = σ(W_Xr X'_t + W_Hr h_{t-1})
        r_t = torch.sigmoid(self.W_xr(x_boosted) + self.W_hr(h_prev))
        
        # Update gate: z_t = σ(W_Xz X'_t + W_Hz h_{t-1})
        z_t = torch.sigmoid(self.W_xz(x_boosted) + self.W_hz(h_prev))
        
        # Candidate: h̃_t = tanh(W_R (r_t ⊙ h_{t-1}) + W_X X'_t)
        h_tilde = torch.tanh(self.W_r(r_t * h_prev) + self.W_x(x_boosted))
        
        # Final: h_t = z_t ⊙ h̃_t + (1 - z_t) ⊙ h_{t-1}
        h_t = z_t * h_tilde + (1 - z_t) * h_prev
        
        return h_t, importance


class AGRU(nn.Module):
    """
    A-GRU Layer: Amygdala-Boosted GRU
    전체 시퀀스에 대해 A-GRU Cell 적용
    """
    
    def __init__(self, input_size: int, hidden_size: int, gamma: float = 1.0):
        super().__init__()
        self.cell = AGRUCell(input_size, hidden_size, gamma)
        self.hidden_size = hidden_size
    
    def forward(
        self, 
        x: torch.Tensor, 
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
            h0: (batch, hidden_size) - 초기 은닉 상태
        
        Returns:
            outputs: (batch, seq_len, hidden_size) - 모든 타임스텝의 은닉 상태
            h_n: (batch, hidden_size) - 마지막 은닉 상태
            importances: (batch, seq_len, input_size) - 각 타임스텝의 중요도
        """
        batch_size, seq_len, _ = x.size()
        
        # 초기 은닉 상태
        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0
        
        outputs = []
        importances = []
        
        for t in range(seq_len):
            h_t, importance = self.cell(x[:, t], h_t)
            outputs.append(h_t.unsqueeze(1))
            importances.append(importance.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        importances = torch.cat(importances, dim=1)
        
        return outputs, h_t, importances


class StackedAGRU(nn.Module):
    """
    Stacked A-GRU: 다층 A-GRU
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 2,
        gamma: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_layers = num_layers
        
        self.agru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.agru_layers.append(AGRU(layer_input_size, hidden_size, gamma))
            if i < num_layers - 1:
                self.dropout_layers.append(nn.Dropout(dropout))
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            outputs: (batch, seq_len, hidden_size) - 마지막 레이어 출력
            h_n: (batch, hidden_size) - 마지막 은닉 상태
            all_importances: list of (batch, seq_len, input_size) - 각 레이어의 중요도
        """
        all_importances = []
        
        current_input = x
        for i, agru in enumerate(self.agru_layers):
            outputs, h_n, importances = agru(current_input)
            all_importances.append(importances)
            
            if i < self.num_layers - 1:
                current_input = self.dropout_layers[i](outputs)
            else:
                current_input = outputs
        
        return outputs, h_n, all_importances


if __name__ == "__main__":
    print("🧪 A-GRU 테스트...")
    
    batch_size, seq_len, input_size = 4, 43, 96
    hidden_size = 64
    
    # 더미 데이터
    x = torch.randn(batch_size, seq_len, input_size)
    
    # 1. A-GRU Cell 테스트
    print("\n1️⃣ A-GRU Cell")
    cell = AGRUCell(input_size, hidden_size, gamma=1.0)
    h_prev = torch.zeros(batch_size, hidden_size)
    h_t, importance = cell(x[:, 0], h_prev)
    print(f"   h_t: {h_t.shape}, importance: {importance.shape}")
    print(f"   Importance range: [{importance.min():.3f}, {importance.max():.3f}]")
    
    # 2. A-GRU Layer 테스트
    print("\n2️⃣ A-GRU Layer")
    agru = AGRU(input_size, hidden_size, gamma=1.0)
    outputs, h_n, importances = agru(x)
    print(f"   outputs: {outputs.shape}")
    print(f"   h_n: {h_n.shape}")
    print(f"   importances: {importances.shape}")
    
    # 3. Stacked A-GRU 테스트
    print("\n3️⃣ Stacked A-GRU (2 layers)")
    stacked_agru = StackedAGRU(input_size, hidden_size, num_layers=2, gamma=1.0, dropout=0.3)
    outputs, h_n, all_importances = stacked_agru(x)
    print(f"   outputs: {outputs.shape}")
    print(f"   h_n: {h_n.shape}")
    print(f"   importances (layer 1): {all_importances[0].shape}")
    print(f"   importances (layer 2): {all_importances[1].shape}")
    
    # 4. 파라미터 비교
    print("\n4️⃣ 파라미터 비교")
    
    # 표준 GRU
    standard_gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
    gru_params = sum(p.numel() for p in standard_gru.parameters())
    
    # A-GRU
    agru_params = sum(p.numel() for p in stacked_agru.parameters())
    
    # A-Net만의 추가 파라미터
    a_net_params = sum(p.numel() for layer in stacked_agru.agru_layers 
                       for p in layer.cell.a_net.parameters())
    
    print(f"   Standard GRU: {gru_params:,} params")
    print(f"   A-GRU:        {agru_params:,} params")
    print(f"   A-Net only:   {a_net_params:,} params (+{a_net_params/gru_params*100:.1f}%)")
    print(f"   Overhead:     +{(agru_params - gru_params):,} params")
    
    print("\n✅ A-GRU 테스트 완료!")





