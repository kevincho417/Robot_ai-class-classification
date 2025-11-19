import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import freeze_support
from torch.cuda.amp import autocast
from pathlib import Path

try:
    from models import (
        EfficientNetWithConvStem,
        EfficientNetWithMultiScale,
        EfficientNetWithResidualStem,
        EfficientNetWithDeepStem,
        EfficientNetWithAttentionStem
    )
except ImportError:
    exit()

TRAIN_DIR = 'dataset2025/train'
TEST_DIR = 'dataset2025/test/unknown'
RUNS_DIR = 'runs'
IMG_SIZE = 224
RESIZE_DIM = 256
BATCH_SIZE = 64
NUM_WORKERS = 4

MODELS_TO_INFER = [
    'conv_stem',
    'multiscale',
    'residual',
    'deep_stem',
    'attention',
]

def get_model(model_type, num_classes=3, dropout=0.0):
    model_dict = {
        'conv_stem': ('EfficientNet + Conv Stem', EfficientNetWithConvStem),
        'multiscale': ('EfficientNet + MultiScale', EfficientNetWithMultiScale),
        'residual': ('EfficientNet + Residual Stem', EfficientNetWithResidualStem),
        'deep_stem': ('EfficientNet + Deep Stem', EfficientNetWithDeepStem),
        'attention': ('EfficientNet + Attention', EfficientNetWithAttentionStem),
    }
    
    if model_type not in model_dict:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model_name, model_class = model_dict[model_type]
    model = model_class(num_classes=num_classes, dropout=dropout) 
    
    return model, model_name

def find_latest_run(runs_dir, model_type):
    base_path = Path(runs_dir)
    run_folders = sorted(
        [f for f in base_path.glob(f"{model_type}_*") if f.is_dir()],
        reverse=True 
    )
    
    if not run_folders:
        return None, None
        
    latest_run_dir = run_folders[0]
    model_path = latest_run_dir / 'best_model.pth'
    
    if not model_path.exists():
        return None, None
        
    return model_path, latest_run_dir

class UnknownTestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def get_id_from_path(self, path):
        filename = os.path.basename(path)
        id_str = filename.replace('test_', '').replace('.jpg', '')
        return int(id_str)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = self.get_id_from_path(image_path)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_id

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and device.type == 'cuda'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    inference_transform = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    test_image_paths = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')))
    if not test_image_paths:
        return

    test_dataset = UnknownTestDataset(test_image_paths, transform=inference_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    try:
        train_dataset_info = datasets.ImageFolder(TRAIN_DIR)
        num_classes = len(train_dataset_info.classes)
    except Exception:
        return

    results_summary = {}

    for model_type in MODELS_TO_INFER:
        model_path, run_dir = find_latest_run(RUNS_DIR, model_type)
        
        if not model_path or not run_dir:
            continue

        model, model_name = get_model(model_type, num_classes=num_classes)

        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception:
            continue

        model = model.to(device)
        model.eval()

        model_results = []
        with torch.no_grad():
            for images, image_ids in tqdm(test_loader, desc=f"Predicting {model_type}"):
                images = images.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp, dtype=torch.float16):
                    outputs = model(images)
                
                _, predicted_indices = torch.max(outputs, 1)
                
                for img_id, pred_idx in zip(image_ids.cpu().numpy(), predicted_indices.cpu().numpy()):
                    model_results.append({'ID': img_id, 'Target': pred_idx})
        
        if model_results:
            df = pd.DataFrame(model_results)
            df = df.sort_values(by='ID')
            output_csv_path = run_dir / 'submission.csv'
            df.to_csv(output_csv_path, index=False)
            results_summary[model_type] = str(output_csv_path)
            
    if results_summary:
        for model_type, path in results_summary.items():
            print(f"  - [{model_type}]: {path}")

if __name__ == '__main__':
    freeze_support()
    main()