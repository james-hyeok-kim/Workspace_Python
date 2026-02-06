import zipfile
import io
import torch

full_path = '/home/jameskimh/workspace/JiT_Pretrained_Model/jit-b-32/real_checkpoint.pth'

def manual_load_zip_checkpoint(path):
    with zipfile.ZipFile(path, 'r') as z:
        # 1. 내부 파일 목록 출력 (디버깅용)
        file_list = z.namelist()
        print("ZIP 내부 파일 목록:", file_list[:10]) # 너무 많을 수 있으니 상위 10개만
        
        # 2. 'data.pkl' 또는 'archive/data.pkl' 같은 피클 파일을 찾습니다.
        # 보통 PyTorch ZIP 포맷은 data.pkl에 메타데이터가 있습니다.
        pickle_files = [f for f in file_list if f.endswith('.pkl')]
        
        if pickle_files:
            target_pkl = pickle_files[0]
            print(f"[{target_pkl}] 파일을 통해 로드를 시도합니다...")
            with z.open(target_pkl) as f:
                # 800MB 이상의 대용량이면 메모리 부족 주의
                return torch.load(io.BytesIO(f.read()), map_location='cpu')
        else:
            raise ValueError("ZIP 내부에 .pkl 파일을 찾을 수 없습니다. 일반적인 PyTorch 포맷이 아닐 수 있습니다.")

try:
    weights = manual_load_zip_checkpoint(full_path)
    # weights가 dict라면 바로 쓰고, 아니면 적절히 처리
    if isinstance(weights, dict):
        final_weights = weights.get('model', weights.get('state_dict', weights))
    else:
        final_weights = weights
    print("수동 로드 성공!")
except Exception as e:
    print(f"수동 로드 실패: {e}")