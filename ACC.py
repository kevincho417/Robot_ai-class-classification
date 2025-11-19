import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torchvision.datasets as datasets

GROUND_TRUTH_FILE = 'solution_test_dataset2025_for_kaggle.csv'
RUNS_DIR = 'runs'
TRAIN_DIR = 'dataset2025/train'

MODELS_TO_CHECK = [
    'conv_stem',
    'multiscale',
    'residual',
    'deep_stem',
    'attention',
]

def find_latest_run_paths(runs_dir, model_type):
    base_path = Path(runs_dir)
    run_folders = sorted(
        [f for f in base_path.glob(f"{model_type}_*") if f.is_dir()],
        reverse=True
    )
    
    if not run_folders:
        return None, None
        
    latest_run_dir = run_folders[0]
    submission_path = latest_run_dir / 'submission.csv'
    
    if not submission_path.exists():
        return None, None
        
    return submission_path, latest_run_dir

def plot_confusion_matrix(cm_normalized, class_names, title, save_path):
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(title, fontsize=16)
        plt.ylabel('Actual (True) Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error: {e}")

def check_all():
    if not os.path.exists(GROUND_TRUTH_FILE):
        return
        
    try:
        df_gt = pd.read_csv(GROUND_TRUTH_FILE, usecols=['ID', 'Target', 'Usage'])
    except Exception:
        return

    try:
        train_dataset_info = datasets.ImageFolder(TRAIN_DIR)
        class_names = train_dataset_info.classes
        num_classes = len(class_names)
    except Exception:
        return
        
    all_results = []

    for model_type in MODELS_TO_CHECK:
        submission_path, run_dir = find_latest_run_paths(RUNS_DIR, model_type)
        
        if not submission_path:
            continue
            
        try:
            df_pred = pd.read_csv(submission_path)
        except Exception:
            continue
            
        merged_df = pd.merge(df_pred, df_gt, on='ID', how='left', suffixes=('_pred', '_gt'))
        merged_df = merged_df.dropna(subset=['Target_gt'])
        merged_df['Target_pred'] = merged_df['Target_pred'].astype(int)
        merged_df['Target_gt'] = merged_df['Target_gt'].astype(int)

        total_correct = (merged_df['Target_pred'] == merged_df['Target_gt']).sum()
        total_count = len(merged_df)
        overall_acc = total_correct / total_count if total_count > 0 else 0
        
        public_df = merged_df[merged_df['Usage'] == 'Public']
        public_correct = (public_df['Target_pred'] == public_df['Target_gt']).sum()
        public_count = len(public_df)
        public_acc = public_correct / public_count if public_count > 0 else 0
        
        private_df = merged_df[merged_df['Usage'] == 'Private']
        private_correct = (private_df['Target_pred'] == private_df['Target_gt']).sum()
        private_count = len(private_df)
        private_acc = private_correct / private_count if private_count > 0 else 0
        
        all_results.append({
            'Model': model_type,
            'Overall_Acc': overall_acc,
            'Public_Acc': public_acc,
            'Private_Acc': private_acc,
            'Run_Folder': run_dir.name
        })

        true_labels = merged_df['Target_gt']
        pred_labels = merged_df['Target_pred']
        
        cm_normalized = confusion_matrix(
            true_labels, 
            pred_labels, 
            normalize='true',
            labels=np.arange(num_classes)
        )
        
        matrix_title = f"Confusion Matrix (Normalized) - {model_type}\nOverall Accuracy: {overall_acc:.4f}"
        matrix_save_path = run_dir / 'confusion_matrix.png'
        
        plot_confusion_matrix(cm_normalized, class_names, matrix_title, matrix_save_path)
        
    if not all_results:
        return
        
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values(by='Private_Acc', ascending=False)
    
    summary_df['Overall_Acc'] = summary_df['Overall_Acc'].map('{:.6f}'.format)
    summary_df['Public_Acc'] = summary_df['Public_Acc'].map('{:.6f}'.format)
    summary_df['Private_Acc'] = summary_df['Private_Acc'].map('{:.6f}'.format)
    
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    check_all()