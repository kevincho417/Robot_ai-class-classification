import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import numpy as np

IMG_SIZE = 256
DATA_DIR = 'dataset2025/train'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

def unnormalize(tensor_img):
    tensor_img = tensor_img.clone()
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor_img = tensor_img * std.view(3, 1, 1) + mean.view(3, 1, 1)
    tensor_img = torch.clamp(tensor_img, 0, 1)
    return tensor_img

def tensor_to_pil(tensor_img):
    np_img = tensor_img.permute(1, 2, 0).numpy()
    return np_img

def visualize_augmentations(data_dir, transform, num_images=5):
    try:
        categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not categories:
            return
        
        random_category = random.choice(categories)
        category_path = os.path.join(data_dir, random_category)
        
        images = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
        if not images:
            return
            
        random_image_name = random.choice(images)
        image_path = os.path.join(category_path, random_image_name)
        original_image = Image.open(image_path).convert('RGB')
    
    except Exception as e:
        print(f"Error: {e}")
        return

    total_plots = num_images + 1
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, total_plots, 1)
    plt.title("Original")
    plt.imshow(original_image.resize((IMG_SIZE, IMG_SIZE)))
    plt.axis('off')
    
    for i in range(num_images):
        augmented_tensor = transform(original_image)
        unnormalized_tensor = unnormalize(augmented_tensor)
        augmented_np = tensor_to_pil(unnormalized_tensor)
        
        plt.subplot(1, total_plots, i + 2)
        plt.title(f"Augmented #{i+1}")
        plt.imshow(augmented_np)
        plt.axis('off')

    plt.suptitle(f"Data Augmentation Visualization\nCategory: {random_category}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Heiti TC', 'sans-serif'] 
        plt.rcParams['axes.unicode_minus'] = False 
    except Exception:
        pass

    visualize_augmentations(DATA_DIR, train_transform, num_images=5)