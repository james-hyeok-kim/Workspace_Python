full_path = '/home/jameskimh/workspace/JiT_Pretrained_Model/jit-h-16/checkpoint-last.pth'

with open(full_path, 'rb') as f:
    header = f.read(20)
    print(f"File Header: {header}")