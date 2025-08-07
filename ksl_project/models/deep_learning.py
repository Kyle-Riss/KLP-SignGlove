# DeepLearningPipeline 스텁
import torch.nn as nn

class DeepLearningPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: 네트워크 구조 정의
    def forward(self, x):
        # TODO: forward 구현
        return {'class_logits': x}
