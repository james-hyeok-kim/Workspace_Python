import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

# 1. 파일 경로 설정
dir_path = '/home/jameskimh/workspace/JiT_Pretrained_Model/jit-b-32'
target_file = 'real_checkpoint.pth'
# os.path.join을 사용하는 것이 더 안전합니다.
full_path = os.path.join(dir_path, target_file)

# 로드 시도 로직
try:
    # 1. 일반적인 로드 시도
    checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        weights = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        weights = checkpoint.state_dict()
except RuntimeError:
    # 2. JIT 로드 시도 (현재 질문자님의 상황에 유력)
    print("일반 로드 실패. JIT 방식으로 재시도합니다...")
    model = torch.jit.load(full_path, map_location='cpu')
    weights = model.state_dict()

def print_weight_stats(weights, layer_name_filter="weight"):
    """가중치 주요 통계치 요약 출력"""
    print(f"{'Layer Name':<50} | {'Mean':>10} | {'Std':>10} | {'Max':>10}")
    print("-" * 88)
    
    for name, param in weights.items():
        if layer_name_filter in name and len(param.shape) >= 2:
            w = param.detach().cpu().numpy()
            mean_val = np.mean(w)
            std_val = np.std(w)
            max_val = np.max(w)
            min_val = np.min(w)
            
            print(f"{name[:50]:<50} | {mean_val:10.6f} | {std_val:10.6f} | {max_val:10.6f}")


def plot_all_layers_channelwise(weights, layer_name_filter="weight", output_dir='all_layer_distributions'):
    """
    모든 레이어의 '모든 채널'을 박스 플롯으로 시각화하여 저장
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리 생성됨: {output_dir}")

    target_layers = [name for name, param in weights.items() 
                     if layer_name_filter in name and len(param.shape) >= 2]
    
    print(f"총 {len(target_layers)}개의 레이어 작업을 시작합니다. 채널 전체를 그리므로 시간이 다소 소요될 수 있습니다.")

    for name in target_layers:
        param = weights[name].detach().cpu()
        num_channels = param.shape[0]
        reshaped_data = param.view(num_channels, -1).numpy()

        # --- 동적 사이즈 조절 ---
        # 채널 1개당 약 0.2인치의 너비를 할당 (최소 10인치, 최대 100인치 제한)
        dynamic_width = max(10, min(100, num_channels * 0.2))
        plt.figure(figsize=(dynamic_width, 8))
        
        # 박스 플롯 생성
        # whis=[1, 99]는 이상치 범위를 1%~99%로 설정하여 극단적인 값까지 표시 (이미지 느낌)
        sns.boxplot(data=[list(ch) for ch in reshaped_data], 
                    palette="husl", fliersize=1, linewidth=0.5)

        # 수평 가이드라인 (전체 레이어의 Min, Max)
        layer_max = np.max(reshaped_data)
        layer_min = np.min(reshaped_data)
        plt.axhline(y=layer_max, color='red', linestyle='--', linewidth=1, alpha=0.4, label=f'Max: {layer_max:.4f}')
        plt.axhline(y=layer_min, color='blue', linestyle='--', linewidth=1, alpha=0.4, label=f'Min: {layer_min:.4f}')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.2)

        # X축 눈금 조절 (채널이 너무 많으면 5개나 10개 단위로 표시)
        if num_channels > 50:
            step = max(1, num_channels // 20) # 약 20개 정도의 라벨만 표시
            plt.xticks(range(0, num_channels, step), range(0, num_channels, step))

        plt.title(f"Layer: {name} (Total Channels: {num_channels})", fontsize=16)
        plt.xlabel("Channel Index")
        plt.ylabel("Weight Value")
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        
        # 파일 저장
        safe_name = name.replace('.', '_')
        save_path = os.path.join(output_dir, f"{safe_name}_all_channels.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150) # 해상도를 높여서 세밀하게 볼 수 있게 함
        plt.close()
        
        print(f"저장 완료: {save_path} (Channels: {num_channels})")

    print(f"\n[완료] 모든 레이어의 전체 채널 분포가 '{output_dir}'에 저장되었습니다.")


def plot_all_layers_3d(weights, layer_name_filter="weight", output_dir='layer_3d_plots'):
    """
    모든 레이어의 가중치를 3D Surface Plot으로 시각화하여 저장
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리 생성됨: {output_dir}")

    target_layers = [name for name, param in weights.items() 
                     if layer_name_filter in name and len(param.shape) >= 2]

    for name in target_layers:
        param = weights[name].detach().cpu()
        
        # 4D Conv 레이어인 경우 (Out, In, H, W) -> (Out, In * H * W) 형태로 평탄화하거나
        # 단순히 (Out, In)의 평균값을 사용합니다. 여기서는 전체 채널 구조를 위해 2D로 변환합니다.
        if len(param.shape) == 4:
            # 커널(H, W)의 중앙값 혹은 평균을 사용하여 Out x In 평면으로 만듭니다.
            data = param.mean(dim=(2, 3)).numpy()
        else:
            data = param.numpy()

        out_dim, in_dim = data.shape

        # 좌표 격자 생성 (모든 채널 포함)
        x = np.arange(in_dim)
        y = np.arange(out_dim)
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 3D 표면도 그리기
        # cmap='coolwarm'은 양수는 빨간색, 음수는 파란색 계열로 표시합니다.
        surf = ax.plot_surface(X, Y, data, cmap='coolwarm', 
                               linewidth=0, antialiased=False, alpha=0.8)

        # 축 설정
        ax.set_title(f"3D Weight Distribution: {name}", fontsize=15)
        ax.set_xlabel('Input Channel')
        ax.set_ylabel('Output Channel')
        ax.set_zlabel('Weight Value')

        # 컬러바 추가
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

        # 시점 조절 (이미지와 유사한 각도)
        ax.view_init(elev=30, azim=-60)

        # 파일 저장
        safe_name = name.replace('.', '_')
        save_path = os.path.join(output_dir, f"{safe_name}_3d.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"3D 그래프 저장 완료: {save_path} (Shape: {data.shape})")

def plot_combined_layers_boxplot(weights, layer_name_filter="weight", output_dir='all_layer_comparison', output_filename='all_layers_comparison.png'):
    """
    모든 레이어의 가중치 분포를 하나의 이미지(박스 플롯)로 통합 시각화
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"디렉토리 생성됨: {output_dir}")



    all_data = []
    layer_names = []

    # 1. 데이터 수집: 가중치 레이어만 필터링하여 리스트에 담기
    for name, param in weights.items():
        if layer_name_filter in name and len(param.shape) >= 2:
            # 해당 레이어의 모든 채널 데이터를 하나로 평탄화
            data = param.detach().cpu().numpy().flatten()
            all_data.append(data)
            # 이름이 너무 길면 끝부분만 보이게 처리 (예: ...layer1.weight)
            short_name = name.split('.')[-2] if len(name.split('.')) > 1 else name
            layer_names.append(name)

    num_layers = len(all_data)
    if num_layers == 0:
        print("시각화할 레이어가 없습니다.")
        return

    # 2. 그래프 크기 설정: 레이어 개수에 따라 가로 길이를 조절
    plt.figure(figsize=(max(15, num_layers * 0.6), 10))
    
    # 3. 통합 박스 플롯 생성
    # showfliers=True: 이상치를 점으로 표현
    bplot = plt.boxplot(all_data, 
                        patch_artist=True, 
                        showfliers=True,
                        # 박스 디자인
                        boxprops=dict(facecolor='#EBF5FB', color='#2980B9', linewidth=1),
                        # 중앙값(Median) 선
                        medianprops=dict(color='#E67E22', linewidth=2),
                        # 이상치(Outlier) 점 디자인
                        flierprops=dict(marker='o', markersize=2, markerfacecolor='red', markeredgecolor='none', alpha=0.3),
                        widths=0.6)

    # 4. 스타일링 및 축 설정
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    plt.title("Weight Distribution Comparison by All Layers", fontsize=22, pad=20)
    plt.ylabel("Weight Value", fontsize=16)
    plt.xlabel("Layer Names", fontsize=16)
    
    # X축 레이블 설정 (레이어 이름을 90도 회전시켜 겹치지 않게 함)
    plt.xticks(range(1, num_layers + 1), layer_names, rotation=90, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 5. 여백 조정 및 저장
    plt.tight_layout()

    # 파일 저장
    save_path = os.path.join(output_dir, f"{output_filename}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # 해상도를 높여서 세밀하게 볼 수 있게 함
    plt.close()
    print(f"통합 시각화 완료: {output_filename} (총 {num_layers}개 레이어)")

# --- 실행 영역 ---

# print("\n=== Weight Statistics Summary ===")
# print_weight_stats(weights)

# 모든 레이어에 대해 실행
#plot_all_layers_channelwise(weights)

#plot_all_layers_3d(weights)

plot_combined_layers_boxplot(weights)