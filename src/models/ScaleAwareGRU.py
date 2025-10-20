"""
Scale-Aware GRU Implementation
각 Multi-Scale CNN 타워의 특징에 독립적인 가중치를 할당하는 GRU 셀
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ScaleAwareGRUCell(nn.Module):
    """
    Scale-Aware GRU Cell
    
    기존 GRU와의 차이점:
    - 입력을 3개 스케일(t3, t5, t7)로 분리
    - 각 스케일에 독립적인 가중치 행렬 할당
    - Update/Reset 게이트가 스케일별 중요도를 학습
    
    수식:
        z_t = sigmoid(W_z3*t3 + W_z5*t5 + W_z7*t7 + U_z*h_{t-1} + b_z)
        r_t = sigmoid(W_r3*t3 + W_r5*t5 + W_r7*t7 + U_r*h_{t-1} + b_r)
        h_tilde = tanh(W_h3*t3 + W_h5*t5 + W_h7*t7 + U_h*(r_t ⊙ h_{t-1}) + b_h)
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
    """
    
    def __init__(
        self,
        scale_sizes: Tuple[int, int, int] = (32, 32, 32),  # (t3, t5, t7)
        hidden_size: int = 64,
        use_hard_functions: bool = False
    ):
        super().__init__()
        
        self.scale_sizes = scale_sizes
        self.hidden_size = hidden_size
        self.use_hard = use_hard_functions
        
        # Update gate (z_t) - 3개 스케일별 가중치
        self.W_z3 = nn.Linear(scale_sizes[0], hidden_size, bias=False)
        self.W_z5 = nn.Linear(scale_sizes[1], hidden_size, bias=False)
        self.W_z7 = nn.Linear(scale_sizes[2], hidden_size, bias=False)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Reset gate (r_t) - 3개 스케일별 가중치
        self.W_r3 = nn.Linear(scale_sizes[0], hidden_size, bias=False)
        self.W_r5 = nn.Linear(scale_sizes[1], hidden_size, bias=False)
        self.W_r7 = nn.Linear(scale_sizes[2], hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Hidden state candidate (h_tilde) - 3개 스케일별 가중치
        self.W_h3 = nn.Linear(scale_sizes[0], hidden_size, bias=False)
        self.W_h5 = nn.Linear(scale_sizes[1], hidden_size, bias=False)
        self.W_h7 = nn.Linear(scale_sizes[2], hidden_size, bias=False)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        t3: torch.Tensor,  # (batch, 32)
        t5: torch.Tensor,  # (batch, 32)
        t7: torch.Tensor,  # (batch, 32)
        h_prev: torch.Tensor  # (batch, hidden_size)
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            t3: Tower 1 출력 (kernel=3)
            t5: Tower 2 출력 (kernel=5)
            t7: Tower 3 출력 (kernel=7)
            h_prev: 이전 은닉 상태
            
        Returns:
            h_t: 현재 은닉 상태 (batch, hidden_size)
        """
        
        # Update gate (z_t)
        z_input = self.W_z3(t3) + self.W_z5(t5) + self.W_z7(t7) + self.U_z(h_prev)
        z_t = F.hardsigmoid(z_input) if self.use_hard else torch.sigmoid(z_input)
        
        # Reset gate (r_t)
        r_input = self.W_r3(t3) + self.W_r5(t5) + self.W_r7(t7) + self.U_r(h_prev)
        r_t = F.hardsigmoid(r_input) if self.use_hard else torch.sigmoid(r_input)
        
        # Hidden state candidate (h_tilde)
        h_input = self.W_h3(t3) + self.W_h5(t5) + self.W_h7(t7) + self.U_h(r_t * h_prev)
        h_tilde = F.hardtanh(h_input) if self.use_hard else torch.tanh(h_input)
        
        # Final hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
    
    def get_gate_weights(self) -> dict:
        """
        각 스케일의 가중치 크기 반환 (해석 가능성)
        
        Returns:
            dict: 각 게이트의 스케일별 가중치 norm
        """
        return {
            'update_gate': {
                'scale_3': self.W_z3.weight.norm().item(),
                'scale_5': self.W_z5.weight.norm().item(),
                'scale_7': self.W_z7.weight.norm().item(),
            },
            'reset_gate': {
                'scale_3': self.W_r3.weight.norm().item(),
                'scale_5': self.W_r5.weight.norm().item(),
                'scale_7': self.W_r7.weight.norm().item(),
            },
            'hidden_gate': {
                'scale_3': self.W_h3.weight.norm().item(),
                'scale_5': self.W_h5.weight.norm().item(),
                'scale_7': self.W_h7.weight.norm().item(),
            }
        }


class ScaleAwareGRU(nn.Module):
    """
    Scale-Aware GRU Layer
    
    ScaleAwareGRUCell을 시퀀스 전체에 적용
    """
    
    def __init__(
        self,
        scale_sizes: Tuple[int, int, int] = (32, 32, 32),
        hidden_size: int = 64,
        use_hard_functions: bool = False
    ):
        super().__init__()
        
        self.scale_sizes = scale_sizes
        self.hidden_size = hidden_size
        
        self.cell = ScaleAwareGRUCell(
            scale_sizes=scale_sizes,
            hidden_size=hidden_size,
            use_hard_functions=use_hard_functions
        )
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, sum(scale_sizes))
        h0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for entire sequence
        
        Args:
            x: 입력 시퀀스 (batch, seq_len, 96)
               - 96 = 32 (t3) + 32 (t5) + 32 (t7)
            h0: 초기 은닉 상태 (batch, hidden_size)
        
        Returns:
            outputs: 모든 타임스텝의 은닉 상태 (batch, seq_len, hidden_size)
            h_n: 마지막 은닉 상태 (batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # 초기 은닉 상태
        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t = h0
        
        outputs = []
        
        # 각 타임스텝마다 GRU 셀 실행
        for t in range(seq_len):
            # 입력을 3개 스케일로 분리
            x_t = x[:, t, :]  # (batch, 96)
            t3 = x_t[:, :self.scale_sizes[0]]  # (batch, 32)
            t5 = x_t[:, self.scale_sizes[0]:self.scale_sizes[0]+self.scale_sizes[1]]  # (batch, 32)
            t7 = x_t[:, self.scale_sizes[0]+self.scale_sizes[1]:]  # (batch, 32)
            
            # GRU 셀 실행
            h_t = self.cell(t3, t5, t7, h_t)
            outputs.append(h_t.unsqueeze(1))
        
        # 모든 타임스텝 결합
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_size)
        
        return outputs, h_t
    
    def get_gate_weights(self) -> dict:
        """각 스케일의 가중치 크기 반환"""
        return self.cell.get_gate_weights()


# Test code
if __name__ == "__main__":
    print("🧪 Scale-Aware GRU 테스트...")
    
    # 파라미터
    batch_size = 4
    seq_len = 43
    scale_sizes = (32, 32, 32)
    hidden_size = 64
    
    # 입력 데이터
    x = torch.randn(batch_size, seq_len, sum(scale_sizes))
    
    print(f"\n입력 shape: {x.shape}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - input_size: {sum(scale_sizes)} (32+32+32)")
    
    # 1. 일반 Sigmoid/Tanh 버전
    print("\n" + "="*70)
    print("1️⃣ Scale-Aware GRU (Sigmoid/Tanh)")
    print("="*70)
    
    gru_normal = ScaleAwareGRU(
        scale_sizes=scale_sizes,
        hidden_size=hidden_size,
        use_hard_functions=False
    )
    
    outputs_normal, h_n_normal = gru_normal(x)
    print(f"✅ 출력 shape: {outputs_normal.shape}")
    print(f"✅ 마지막 은닉 상태: {h_n_normal.shape}")
    
    # 가중치 분석
    weights = gru_normal.get_gate_weights()
    print(f"\n📊 Update Gate 가중치 크기:")
    print(f"  - Scale 3 (kernel=3): {weights['update_gate']['scale_3']:.4f}")
    print(f"  - Scale 5 (kernel=5): {weights['update_gate']['scale_5']:.4f}")
    print(f"  - Scale 7 (kernel=7): {weights['update_gate']['scale_7']:.4f}")
    
    # 2. Hard 함수 버전
    print("\n" + "="*70)
    print("2️⃣ Scale-Aware GRU (HardSigmoid/HardTanh)")
    print("="*70)
    
    gru_hard = ScaleAwareGRU(
        scale_sizes=scale_sizes,
        hidden_size=hidden_size,
        use_hard_functions=True
    )
    
    outputs_hard, h_n_hard = gru_hard(x)
    print(f"✅ 출력 shape: {outputs_hard.shape}")
    print(f"✅ 마지막 은닉 상태: {h_n_hard.shape}")
    
    # 3. 파라미터 수 비교
    print("\n" + "="*70)
    print("3️⃣ 파라미터 수 비교")
    print("="*70)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 기존 GRU (비교용)
    gru_standard = nn.GRU(sum(scale_sizes), hidden_size, 1, batch_first=True)
    
    print(f"기존 GRU:          {count_parameters(gru_standard):,} 파라미터")
    print(f"Scale-Aware GRU:   {count_parameters(gru_normal):,} 파라미터")
    print(f"증가율:            {count_parameters(gru_normal) / count_parameters(gru_standard):.2f}x")
    
    # 4. 속도 테스트
    print("\n" + "="*70)
    print("4️⃣ 추론 속도 비교")
    print("="*70)
    
    import time
    
    # Warmup
    for _ in range(10):
        _ = gru_normal(x)
        _ = gru_hard(x)
    
    # 일반 버전
    start = time.time()
    for _ in range(100):
        _ = gru_normal(x)
    time_normal = (time.time() - start) / 100 * 1000
    
    # Hard 버전
    start = time.time()
    for _ in range(100):
        _ = gru_hard(x)
    time_hard = (time.time() - start) / 100 * 1000
    
    print(f"Sigmoid/Tanh:      {time_normal:.2f}ms")
    print(f"HardSigmoid/Tanh:  {time_hard:.2f}ms")
    print(f"속도 향상:         {(time_normal - time_hard) / time_normal * 100:.1f}%")
    
    print("\n✅ 모든 테스트 통과!")

