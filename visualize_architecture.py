#!/usr/bin/env python3
"""
MS-CSGRU 모델 아키텍처 시각화 스크립트
이미지 스타일의 플로우차트 생성
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_architecture_flowchart():
    """MS-CSGRU 아키텍처 플로우차트 생성"""
    
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    ax.axis('off')
    
    # 색상 정의
    color_input = '#2E4053'
    color_cnn = '#1F618D'
    color_gru = '#117864'
    color_attention = '#7D3C98'
    color_output = '#B9770E'
    color_special = '#922B21'
    
    y_pos = 24
    
    # ============================================================
    # 1. 입력 레이어
    # ============================================================
    input_box = FancyBboxPatch(
        (3, y_pos-0.8), 4, 0.6,
        boxstyle="round,pad=0.1",
        facecolor=color_input,
        edgecolor='white',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(5, y_pos-0.5, '음성 입력\n센서 데이터 입력', 
            ha='center', va='center', fontsize=11, color='white', weight='bold')
    
    y_pos -= 1.2
    
    # 입력 상세 정보
    detail_box = FancyBboxPatch(
        (2.5, y_pos-0.8), 5, 0.6,
        boxstyle="round,pad=0.05",
        facecolor='#34495E',
        edgecolor='white',
        linewidth=1
    )
    ax.add_patch(detail_box)
    ax.text(5, y_pos-0.5, '텍스트 단위 입력: 320ms chunks\n센서 특징 [T, 8] = [87, 8]', 
            ha='center', va='center', fontsize=9, color='white')
    
    y_pos -= 1.5
    
    # 화살표
    arrow = FancyArrowPatch(
        (5, y_pos+0.5), (5, y_pos),
        arrowstyle='->', mutation_scale=30, linewidth=2, color='white'
    )
    ax.add_patch(arrow)
    
    y_pos -= 0.5
    
    # ============================================================
    # 2. Multi-Scale CNN Encoder
    # ============================================================
    
    # 큰 박스 (전체 CNN 영역)
    cnn_box = FancyBboxPatch(
        (0.5, y_pos-5.5), 9, 5,
        boxstyle="round,pad=0.1",
        facecolor=color_cnn,
        edgecolor='white',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(cnn_box)
    ax.text(5, y_pos-0.3, 'Multi-Scale CNN Encoder', 
            ha='center', va='center', fontsize=12, color='white', weight='bold')
    
    y_pos -= 1
    
    # Multi-Head Attention 라벨
    ax.text(5, y_pos-0.2, 'Multi-Head Attention (병렬 처리)', 
            ha='center', va='center', fontsize=10, color='white', style='italic')
    
    y_pos -= 0.5
    
    # 3개 타워 (병렬)
    tower_width = 2.2
    tower_height = 2
    tower_spacing = 0.4
    
    towers = [
        {'name': 'Tower 1', 'kernel': 3, 'padding': 1, 'x': 1.5},
        {'name': 'Tower 2', 'kernel': 5, 'padding': 2, 'x': 4},
        {'name': 'Tower 3', 'kernel': 7, 'padding': 3, 'x': 6.5}
    ]
    
    for tower in towers:
        # 타워 박스
        tower_box = FancyBboxPatch(
            (tower['x'], y_pos-tower_height), tower_width, tower_height,
            boxstyle="round,pad=0.05",
            facecolor='#1A5490',
            edgecolor='white',
            linewidth=1.5
        )
        ax.add_patch(tower_box)
        
        # 타워 내용
        tower_y = y_pos - 0.3
        ax.text(tower['x'] + tower_width/2, tower_y, tower['name'], 
                ha='center', va='center', fontsize=10, color='white', weight='bold')
        tower_y -= 0.4
        ax.text(tower['x'] + tower_width/2, tower_y, f"Conv1D(k={tower['kernel']})", 
                ha='center', va='center', fontsize=9, color='white')
        tower_y -= 0.3
        ax.text(tower['x'] + tower_width/2, tower_y, 'in: 8', 
                ha='center', va='center', fontsize=8, color='#D5DBDB')
        tower_y -= 0.25
        ax.text(tower['x'] + tower_width/2, tower_y, 'out: 32', 
                ha='center', va='center', fontsize=8, color='#D5DBDB')
        tower_y -= 0.25
        ax.text(tower['x'] + tower_width/2, tower_y, f"padding: {tower['padding']}", 
                ha='center', va='center', fontsize=8, color='#D5DBDB')
        tower_y -= 0.35
        ax.text(tower['x'] + tower_width/2, tower_y, 'BatchNorm → ReLU', 
                ha='center', va='center', fontsize=8, color='#AED6F1')
    
    y_pos -= tower_height + 0.3
    
    # 3개 타워에서 Concatenate로 화살표
    for tower in towers:
        arrow = FancyArrowPatch(
            (tower['x'] + tower_width/2, y_pos+0.3), (5, y_pos),
            arrowstyle='->', mutation_scale=20, linewidth=1.5, color='white'
        )
        ax.add_patch(arrow)
    
    y_pos -= 0.5
    
    # Concatenate
    concat_box = FancyBboxPatch(
        (3.5, y_pos-0.5), 3, 0.4,
        boxstyle="round,pad=0.05",
        facecolor='#1A5490',
        edgecolor='white',
        linewidth=1.5
    )
    ax.add_patch(concat_box)
    ax.text(5, y_pos-0.3, 'Concatenate (96 channels)', 
            ha='center', va='center', fontsize=9, color='white', weight='bold')
    
    y_pos -= 0.8
    
    # 후처리
    post_box = FancyBboxPatch(
        (3, y_pos-0.8), 4, 0.6,
        boxstyle="round,pad=0.05",
        facecolor='#1A5490',
        edgecolor='white',
        linewidth=1.5
    )
    ax.add_patch(post_box)
    ax.text(5, y_pos-0.5, 'BatchNorm → ReLU → MaxPool(2) → Dropout(0.3)', 
            ha='center', va='center', fontsize=9, color='white')
    
    y_pos -= 1.2
    
    # 히든 상태
    hidden_box = FancyBboxPatch(
        (3.5, y_pos-0.5), 3, 0.4,
        boxstyle="round,pad=0.05",
        facecolor='#34495E',
        edgecolor='white',
        linewidth=1
    )
    ax.add_patch(hidden_box)
    ax.text(5, y_pos-0.3, '히든 상태 H [T/2, 96] = [43, 96]', 
            ha='center', va='center', fontsize=9, color='white')
    
    y_pos -= 1.2
    
    # 화살표
    arrow = FancyArrowPatch(
        (5, y_pos+0.5), (5, y_pos),
        arrowstyle='->', mutation_scale=30, linewidth=2, color='white'
    )
    ax.add_patch(arrow)
    
    y_pos -= 0.5
    
    # ============================================================
    # 3. Stacked GRU Layers
    # ============================================================
    
    # 큰 박스 (전체 GRU 영역)
    gru_box = FancyBboxPatch(
        (0.5, y_pos-4), 9, 3.5,
        boxstyle="round,pad=0.1",
        facecolor=color_gru,
        edgecolor='white',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(gru_box)
    ax.text(5, y_pos-0.3, 'Stacked GRU Layers', 
            ha='center', va='center', fontsize=12, color='white', weight='bold')
    
    y_pos -= 0.8
    
    # GRU Layer 1
    gru1_box = FancyBboxPatch(
        (2, y_pos-1.2), 6, 1,
        boxstyle="round,pad=0.05",
        facecolor='#148F77',
        edgecolor='white',
        linewidth=1.5
    )
    ax.add_patch(gru1_box)
    ax.text(5, y_pos-0.3, 'GRU Layer 1 (First Layer)', 
            ha='center', va='center', fontsize=10, color='white', weight='bold')
    ax.text(5, y_pos-0.6, 'GRU Cell (input: 96, hidden: 64)', 
            ha='center', va='center', fontsize=9, color='white')
    ax.text(5, y_pos-0.9, 'Dropout(p=0.3) → 출력: [43, 64]', 
            ha='center', va='center', fontsize=8, color='#D5DBDB')
    
    y_pos -= 1.5
    
    # 화살표
    arrow = FancyArrowPatch(
        (5, y_pos+0.3), (5, y_pos),
        arrowstyle='->', mutation_scale=25, linewidth=2, color='white'
    )
    ax.add_patch(arrow)
    
    y_pos -= 0.3
    
    # GRU Layer 2
    gru2_box = FancyBboxPatch(
        (2, y_pos-1.2), 6, 1,
        boxstyle="round,pad=0.05",
        facecolor='#148F77',
        edgecolor='white',
        linewidth=1.5
    )
    ax.add_patch(gru2_box)
    ax.text(5, y_pos-0.3, 'GRU Layer 2 (Second Layer)', 
            ha='center', va='center', fontsize=10, color='white', weight='bold')
    ax.text(5, y_pos-0.6, 'GRU Cell (input: 64, hidden: 64)', 
            ha='center', va='center', fontsize=9, color='white')
    ax.text(5, y_pos-0.9, '출력: [43, 64]', 
            ha='center', va='center', fontsize=8, color='#D5DBDB')
    
    y_pos -= 1.8
    
    # 화살표
    arrow = FancyArrowPatch(
        (5, y_pos+0.5), (5, y_pos),
        arrowstyle='->', mutation_scale=30, linewidth=2, color='white'
    )
    ax.add_patch(arrow)
    
    y_pos -= 0.5
    
    # ============================================================
    # 4. 패딩 인식 특징 추출 (핵심!)
    # ============================================================
    
    padding_box = FancyBboxPatch(
        (1.5, y_pos-1.8), 7, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=color_special,
        edgecolor='white',
        linewidth=2
    )
    ax.add_patch(padding_box)
    ax.text(5, y_pos-0.3, '패딩 인식 특징 추출 (Padding-Aware) ⭐', 
            ha='center', va='center', fontsize=11, color='white', weight='bold')
    
    # 코드 스타일로 표시
    code_text = """if x_padding is not None:
    valid_lengths = (x_padding == 0).sum(dim=1) - 1
    final_features = gru_out[batch_idx, valid_lengths]
else:
    final_features = gru_out[:, -1, :]"""
    
    ax.text(5, y_pos-1, code_text, 
            ha='center', va='center', fontsize=7, color='#F8F9F9',
            family='monospace', bbox=dict(boxstyle='round', facecolor='#1C2833', alpha=0.8))
    
    y_pos -= 2.2
    
    # 출력 차원
    dim_box = FancyBboxPatch(
        (3.5, y_pos-0.5), 3, 0.4,
        boxstyle="round,pad=0.05",
        facecolor='#34495E',
        edgecolor='white',
        linewidth=1
    )
    ax.add_patch(dim_box)
    ax.text(5, y_pos-0.3, '출력: [batch, 64]', 
            ha='center', va='center', fontsize=9, color='white')
    
    y_pos -= 1
    
    # 화살표
    arrow = FancyArrowPatch(
        (5, y_pos+0.5), (5, y_pos),
        arrowstyle='->', mutation_scale=30, linewidth=2, color='white'
    )
    ax.add_patch(arrow)
    
    y_pos -= 0.5
    
    # ============================================================
    # 5. Classifier (분류기)
    # ============================================================
    
    classifier_box = FancyBboxPatch(
        (1.5, y_pos-2.5), 7, 2,
        boxstyle="round,pad=0.1",
        facecolor=color_output,
        edgecolor='white',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(classifier_box)
    ax.text(5, y_pos-0.3, 'Classifier (분류기)', 
            ha='center', va='center', fontsize=12, color='white', weight='bold')
    
    y_pos -= 0.7
    
    # 분류기 레이어들
    classifier_layers = [
        'Linear(64 → 128)',
        'ReLU',
        'Dropout(p=0.3)',
        'Linear(128 → 24)',
        '로짓 출력 [batch, 24]'
    ]
    
    for i, layer in enumerate(classifier_layers):
        if i < len(classifier_layers) - 1:
            layer_box = FancyBboxPatch(
                (3, y_pos-0.35), 4, 0.3,
                boxstyle="round,pad=0.03",
                facecolor='#CA6F1E',
                edgecolor='white',
                linewidth=1
            )
            ax.add_patch(layer_box)
            ax.text(5, y_pos-0.2, layer, 
                    ha='center', va='center', fontsize=9, color='white')
            y_pos -= 0.4
            
            if i < len(classifier_layers) - 2:
                arrow = FancyArrowPatch(
                    (5, y_pos+0.05), (5, y_pos),
                    arrowstyle='->', mutation_scale=20, linewidth=1.5, color='white'
                )
                ax.add_patch(arrow)
                y_pos -= 0.05
        else:
            # 마지막 출력
            output_box = FancyBboxPatch(
                (3.5, y_pos-0.4), 3, 0.35,
                boxstyle="round,pad=0.05",
                facecolor='#34495E',
                edgecolor='white',
                linewidth=1
            )
            ax.add_patch(output_box)
            ax.text(5, y_pos-0.225, layer, 
                    ha='center', va='center', fontsize=9, color='white', weight='bold')
    
    y_pos -= 0.8
    
    # 화살표
    arrow = FancyArrowPatch(
        (5, y_pos+0.4), (5, y_pos),
        arrowstyle='->', mutation_scale=30, linewidth=2, color='white'
    )
    ax.add_patch(arrow)
    
    y_pos -= 0.5
    
    # ============================================================
    # 6. 출력
    # ============================================================
    
    output_box = FancyBboxPatch(
        (2.5, y_pos-1), 5, 0.8,
        boxstyle="round,pad=0.1",
        facecolor=color_output,
        edgecolor='white',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(5, y_pos-0.4, '음성 출력 (Output)', 
            ha='center', va='center', fontsize=11, color='white', weight='bold')
    ax.text(5, y_pos-0.7, '영어 음성 S (24 classes)', 
            ha='center', va='center', fontsize=9, color='white')
    
    y_pos -= 1.5
    
    # 실시간 처리 정보
    realtime_box = FancyBboxPatch(
        (2, y_pos-0.6), 6, 0.5,
        boxstyle="round,pad=0.05",
        facecolor='#34495E',
        edgecolor='white',
        linewidth=1
    )
    ax.add_patch(realtime_box)
    ax.text(5, y_pos-0.35, '실시간 스트리밍 처리: 320ms chunks', 
            ha='center', va='center', fontsize=9, color='white')
    
    # 제목
    fig.suptitle('MS-CSGRU 모델 아키텍처 플로우차트\n(Multi-Scale CNN + Stacked GRU)', 
                 fontsize=16, color='white', weight='bold', y=0.98)
    
    # 배경색
    fig.patch.set_facecolor('#1C2833')
    ax.set_facecolor('#1C2833')
    
    plt.tight_layout()
    return fig

def create_dimension_flow():
    """차원 변화 플로우차트"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    y_pos = 11
    
    stages = [
        {'name': '입력', 'dim': '(batch, 87, 8)', 'color': '#2E4053'},
        {'name': 'transpose', 'dim': '(batch, 8, 87)', 'color': '#34495E'},
        {'name': '3 Towers', 'dim': '(batch, 32, 87) × 3', 'color': '#1F618D'},
        {'name': 'Concat', 'dim': '(batch, 96, 87)', 'color': '#1F618D'},
        {'name': 'MaxPool(2)', 'dim': '(batch, 96, 43)', 'color': '#1F618D'},
        {'name': 'transpose', 'dim': '(batch, 43, 96)', 'color': '#34495E'},
        {'name': 'GRU1', 'dim': '(batch, 43, 64)', 'color': '#117864'},
        {'name': 'GRU2', 'dim': '(batch, 43, 64)', 'color': '#117864'},
        {'name': 'Padding-Aware', 'dim': '(batch, 64)', 'color': '#922B21'},
        {'name': 'Dense', 'dim': '(batch, 24)', 'color': '#B9770E'},
    ]
    
    for i, stage in enumerate(stages):
        # 박스
        box = FancyBboxPatch(
            (2, y_pos-0.6), 6, 0.5,
            boxstyle="round,pad=0.05",
            facecolor=stage['color'],
            edgecolor='white',
            linewidth=2
        )
        ax.add_patch(box)
        
        # 텍스트
        ax.text(3, y_pos-0.35, stage['name'], 
                ha='left', va='center', fontsize=11, color='white', weight='bold')
        ax.text(7, y_pos-0.35, stage['dim'], 
                ha='right', va='center', fontsize=10, color='#D5DBDB', family='monospace')
        
        y_pos -= 0.8
        
        # 화살표
        if i < len(stages) - 1:
            arrow = FancyArrowPatch(
                (5, y_pos+0.2), (5, y_pos),
                arrowstyle='->', mutation_scale=25, linewidth=2, color='white'
            )
            ax.add_patch(arrow)
            y_pos -= 0.2
    
    # 제목
    fig.suptitle('MS-CSGRU 차원 변화 플로우', 
                 fontsize=16, color='white', weight='bold', y=0.98)
    
    # 배경색
    fig.patch.set_facecolor('#1C2833')
    ax.set_facecolor('#1C2833')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("🎨 MS-CSGRU 아키텍처 플로우차트 생성 중...")
    
    # 메인 플로우차트
    fig1 = create_architecture_flowchart()
    fig1.savefig('visualizations/MSCSGRU_architecture_flowchart.png', 
                 dpi=300, bbox_inches='tight', facecolor='#1C2833')
    print("✅ 저장: visualizations/MSCSGRU_architecture_flowchart.png")
    
    # 차원 변화 플로우
    fig2 = create_dimension_flow()
    fig2.savefig('visualizations/MSCSGRU_dimension_flow.png', 
                 dpi=300, bbox_inches='tight', facecolor='#1C2833')
    print("✅ 저장: visualizations/MSCSGRU_dimension_flow.png")
    
    print("\n🎉 플로우차트 생성 완료!")
    print("📂 파일 위치:")
    print("   - visualizations/MSCSGRU_architecture_flowchart.png")
    print("   - visualizations/MSCSGRU_dimension_flow.png")

