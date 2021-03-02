from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader

@dataclass
class FLUser:
    user_id: int = 0
    compromised: bool = False
    # Dataloader包括了选取index的方式Sampler和Dataset中对应的数据
    # 便于pytorch使用并进行训练
    train_loader: DataLoader = None
