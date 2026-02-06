import zipfile
import torch
import os

dir_path = '/home/jameskimh/workspace/JiT_Pretrained_Model/jit-h-16'
source_file = 'checkpoint-last_unzip.pth'
extract_file = 'real_checkpoint.pth'
# os.path.join을 사용하는 것이 더 안전합니다.
full_path = os.path.join(dir_path, source_file)
extract_path = os.path.join(dir_path, extract_file)

try:
    with zipfile.ZipFile(full_path, 'r') as z:
        print("내부 파일 추출 중: 'checkpoint-last.pth' -> 'real_checkpoint.pth'...")
        # ZIP 안에 있는 'checkpoint-last.pth'를 꺼내서 다른 이름으로 저장
        with z.open('checkpoint-last.pth') as source, open(extract_path, 'wb') as target:
            target.write(source.read())
    
    print("추출 완료! 이제 진짜 파일을 로드합니다.")
    
    # 추출된 파일로 다시 로드 시도
    checkpoint = torch.load(extract_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        weights = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        weights = checkpoint
    
    print("가중치 로드 성공! 키 개수:", len(weights.keys()))
    # 이제 이 weights를 가지고 이전에 만든 시각화 함수를 돌리시면 됩니다.
    # plot_weight_distribution(weights)

except Exception as e:
    print(f"실패 원인: {e}")