import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import glob
import pandas as pd
from multiprocessing import freeze_support
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import requests
import matplotlib.font_manager as fm

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    exit()

try:
    from captum.attr import IntegratedGradients
    from captum.attr import visualization as viz
except ImportError:
    exit()

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

GROUND_TRUTH_FILE = 'solution_test_dataset2025_for_kaggle.csv'
TRAIN_DIR = 'dataset2025/train'
TEST_DIR = 'dataset2025/test/unknown'
RUNS_DIR = 'runs'
IMG_SIZE = 224
RESIZE_DIM = 256

MODELS_TO_CHECK = [
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

def find_latest_run_paths(runs_dir, model_type):
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

def get_id_from_path(path):
    filename = os.path.basename(path)
    id_str = filename.replace('test_', '').replace('.jpg', '')
    return int(id_str)

def get_target_images(gt_file, test_dir, train_dir):
    map_id_to_path = {}
    test_image_paths = glob.glob(os.path.join(TEST_DIR, '*.jpg'))
    if not test_image_paths:
        return []
        
    for path in test_image_paths:
        try:
            img_id = get_id_from_path(path)
            map_id_to_path[img_id] = path
        except ValueError:
            pass

    df_gt = pd.read_csv(gt_file)
    class_names = datasets.ImageFolder(train_dir).classes
    
    target_images = []
    
    for idx, name in enumerate(class_names):
        class_rows = df_gt[df_gt['Target'] == idx]
        
        if class_rows.empty:
            continue
            
        image_to_find = 1
        if idx == 1: 
            image_to_find = 2
            
        found_count = 0
        found = False
        for _, row in class_rows.iterrows():
            image_id = row['ID']
            if image_id in map_id_to_path:
                found_count += 1
                if found_count == image_to_find:
                    image_path = map_id_to_path[image_id]
                    target_images.append({
                        'class_name': name,
                        'path': image_path,
                        'target_idx': idx,
                        'id': image_id
                    })
                    found = True
                    break
        
    return target_images

def download_cjk_font():
    FONT_FILENAME = 'NotoSansTC-Regular.otf'
    FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/notosanstc/NotoSansTC-Regular.otf"
    
    if not os.path.exists(FONT_FILENAME):
        try:
            r = requests.get(FONT_URL)
            r.raise_for_status()
            with open(FONT_FILENAME, 'wb') as f:
                f.write(r.content)
        except Exception:
            return False
            
    try:
        fm.fontManager.addfont(FONT_FILENAME)
        plt.rc('font', family='Noto Sans TC')
        plt.rcParams['axes.unicode_minus'] = False 
        return True
    except Exception:
        return False

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_cjk_font()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_model_input = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    transform_visualize = transforms.Compose([
        transforms.Resize(RESIZE_DIM),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])

    target_images = get_target_images(GROUND_TRUTH_FILE, TEST_DIR, TRAIN_DIR)
    if not target_images:
        return

    num_classes = len(datasets.ImageFolder(TRAIN_DIR).classes)
        
    for model_type in MODELS_TO_CHECK:
        model_path, run_dir = find_latest_run_paths(RUNS_DIR, model_type)
        if not model_path:
            continue
            
        model, model_name = get_model(model_type, num_classes=num_classes)
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
        except Exception:
            continue
            
        model = model.to(device).eval()
        target_layers = [model.backbone.conv_head]
        cam = GradCAM(model=model, target_layers=target_layers)
        ig = IntegratedGradients(model)
        
        xai_save_dir = run_dir / 'xai_visualizations'
        xai_save_dir.mkdir(exist_ok=True)

        for image_info in target_images:
            class_name = image_info['class_name']
            img_path = image_info['path']
            target_idx = image_info['target_idx']
            image_id = image_info['id']
            
            pil_img = Image.open(img_path).convert('RGB')
            input_tensor = transform_model_input(pil_img).unsqueeze(0).to(device)
            viz_tensor = transform_visualize(pil_img)
            rgb_img_numpy = viz_tensor.permute(1, 2, 0).numpy()

            try:
                cam_targets = [ClassifierOutputTarget(target_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=cam_targets)[0, :]
                cam_visualization = show_cam_on_image(rgb_img_numpy, grayscale_cam, use_rgb=True)
            except Exception:
                cam_visualization = np.zeros_like(rgb_img_numpy)
                
            try:
                baseline = torch.zeros_like(input_tensor).to(device)
                attributions_ig = ig.attribute(
                    input_tensor, 
                    baselines=baseline, 
                    target=target_idx, 
                    n_steps=50, 
                    internal_batch_size=4
                )
                ig_viz_data = attributions_ig.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            except Exception:
                ig_viz_data = None

            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(f"XAI for {model_type} - Predicting: '{class_name}' (ID: {image_id})", fontsize=20)
            
            axes[0].imshow(rgb_img_numpy)
            axes[0].set_title('Original Image', fontsize=16)
            axes[0].axis('off')
            
            axes[1].imshow(cam_visualization)
            axes[1].set_title('Grad-CAM', fontsize=16)
            axes[1].axis('off')
            
            axes[2].set_title('Integrated Gradients', fontsize=16)
            if ig_viz_data is not None:
                viz.visualize_image_attr(
                    ig_viz_data,
                    rgb_img_numpy,
                    method='blended_heat_map',
                    sign='all',
                    show_colorbar=True,
                    title="",
                    plt_fig_axis=(fig, axes[2]),
                    use_pyplot=False 
                )
            else:
                axes[2].imshow(np.zeros_like(rgb_img_numpy))
                axes[2].text(0.5, 0.5, 'IG Failed', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes, color='white')
            
            save_path = xai_save_dir / f"xai_on_class_{class_name}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

if __name__ == '__main__':
    freeze_support()
    main()