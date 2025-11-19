import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import random
import time
import os
import json
from datetime import datetime
from pathlib import Path
from multiprocessing import freeze_support
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from models import (
        EfficientNetWithConvStem,
        EfficientNetWithMultiScale,
        EfficientNetWithResidualStem,
        EfficientNetWithDeepStem,
        EfficientNetWithAttentionStem
    )
except ImportError:
    from torchvision.models import efficientnet_b0
    class EfficientNetWithConvStem(nn.Module):
        def __init__(self, num_classes=3, dropout=0.3):
            super().__init__()
            self.base = efficientnet_b0(weights='DEFAULT')
            in_features = self.base.classifier[1].in_features
            self.base.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        def forward(self, x):
            return self.base(x)

    EfficientNetWithMultiScale = EfficientNetWithConvStem
    EfficientNetWithResidualStem = EfficientNetWithConvStem
    EfficientNetWithDeepStem = EfficientNetWithConvStem
    EfficientNetWithAttentionStem = EfficientNetWithConvStem

DATA_DIR = 'dataset2025/train'
RUNS_DIR = 'runs'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 150
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 10

MODELS_TO_TRAIN = [
    'conv_stem',
    'multiscale',
    'residual',
    'deep_stem',
    'attention',
]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    normalize,
    transforms.RandomErasing(p=0.3),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize,
])

class TrainingLogger:
    def __init__(self, save_dir, model_name):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        self.info = {
            'model_name': model_name,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_params': 0,
            'trainable_params': 0,
        }
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(float(lr))
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        
        checkpoint_path = self.save_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def plot_learning_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {self.model_name}', fontsize=16)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[0, 1].axhline(y=self.best_val_acc, color='g', linestyle='--', 
                           label=f'Best: {self.best_val_acc:.4f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        train_val_gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
        axes[1, 1].plot(epochs, train_val_gap, 'purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(epochs, 0, train_val_gap, alpha=0.3, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train Acc - Val Acc')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / 'learning_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_summary(self, total_time):
        self.info['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.info['total_time'] = f"{total_time // 60:.0f}m {total_time % 60:.0f}s"
        self.info['best_val_acc'] = float(self.best_val_acc)
        self.info['best_epoch'] = int(self.best_epoch)
        self.info['total_epochs'] = len(self.history['train_loss'])
        
        summary_path = self.save_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'info': self.info,
                'history': self.history
            }, f, indent=4)
        
        txt_path = self.save_dir / 'training_summary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Training Summary - {self.model_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Start Time: {self.info['start_time']}\n")
            f.write(f"End Time: {self.info['end_time']}\n")
            f.write(f"Total Time: {self.info['total_time']}\n\n")
            f.write(f"Total Parameters: {self.info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.info['trainable_params']:,}\n\n")
            f.write(f"Total Epochs: {self.info['total_epochs']}\n")
            f.write(f"Best Epoch: {self.best_epoch + 1}\n")
            f.write(f"Best Val Accuracy: {self.best_val_acc:.4f}\n\n")
            f.write(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Train Acc: {self.history['train_acc'][-1]:.4f}\n")
            f.write(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}\n")
            f.write(f"Final Val Acc: {self.history['val_acc'][-1]:.4f}\n")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dataloaders(data_dir, val_split, train_transform, val_transform, batch_size, num_workers, seed):
    if not os.path.isdir(data_dir):
        dummy_dir = Path(data_dir)
        (dummy_dir / 'class_a').mkdir(parents=True, exist_ok=True)
        (dummy_dir / 'class_b').mkdir(parents=True, exist_ok=True)
        (dummy_dir / 'class_c').mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for c in ['class_a', 'class_b', 'class_c']:
            for i in range(100):
                img = Image.new('RGB', (256, 256), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
                img.save(dummy_dir / c / f'dummy_{i}.png')

    full_train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    full_val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)
    
    class_names = full_train_dataset.classes
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    
    targets = full_train_dataset.targets
    train_indices, val_indices = [], []
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    
    try:
        for train_idx, val_idx in sss.split(indices, targets):
            train_indices = train_idx
            val_indices = val_idx
    except ValueError:
        random.Random(seed).shuffle(indices)
        split_point = int(dataset_size * val_split)
        val_indices = indices[:split_point]
        train_indices = indices[split_point:]

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=True if num_workers > 0 else False)
    
    return train_loader, val_loader, class_names

def get_model(model_type, num_classes=3, dropout=0.3):
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

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False, unit="batch")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=device.type == 'cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        pbar.set_postfix(loss=loss.item())

    pbar.close()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    
    return epoch_loss, epoch_acc.item()

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(val_loader, desc="Validation", leave=False, unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast(enabled=device.type == 'cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            pbar.set_postfix(loss=loss.item())
    
    pbar.close()
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = running_corrects.double() / len(val_loader.dataset)
    
    return epoch_loss, epoch_acc.item()

def train_model(model, model_name, train_loader, val_loader, device, 
                num_epochs, save_dir, early_stop_patience):
    
    logger = TrainingLogger(save_dir, model_name)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info['total_params'] = total_params
    logger.info['trainable_params'] = trainable_params
    
    print(f"Training: {model_name}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Save Directory: {save_dir}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        is_best = val_acc > logger.best_val_acc
        
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        logger.save_checkpoint(model, optimizer, epoch, is_best)
        
        if is_best:
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            break
    
    total_time = time.time() - start_time
    
    logger.plot_learning_curves()
    logger.save_summary(total_time)
    
    return logger.best_val_acc, logger.best_epoch

if __name__ == '__main__':
    freeze_support() 
    set_seed(RANDOM_SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, class_names = get_dataloaders(
        DATA_DIR, VAL_SPLIT, train_transform, val_transform,
        BATCH_SIZE, NUM_WORKERS, RANDOM_SEED
    )
    
    num_classes = len(class_names)
    
    runs_dir = Path(RUNS_DIR)
    runs_dir.mkdir(exist_ok=True)
    
    results = {}
    
    for i, model_type in enumerate(MODELS_TO_TRAIN, 1):
        try:
            model, model_name = get_model(model_type, num_classes, dropout=0.3)
            model = model.to(device)
        except Exception:
            continue
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = runs_dir / f"{model_type}_{timestamp}"
        
        best_acc, best_epoch = train_model(
            model, model_name, train_loader, val_loader, device,
            NUM_EPOCHS, save_dir, EARLY_STOP_PATIENCE
        )
        
        results[model_type] = {
            'model_name': model_name,
            'best_acc': best_acc,
            'best_epoch': best_epoch,
            'save_dir': str(save_dir)
        }
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
    
    for rank, (model_type, info) in enumerate(sorted_results, 1):
        print(f"{rank:<3} {info['model_name']:<28} {info['best_acc']:.4f} {info['best_epoch']+1:<12}")