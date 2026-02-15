#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted Montage Classification for Ablation Study
Matches dataset structure of MIL/Early/Late Fusion experiments.
"""

import os
import sys
import timm
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
#import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report, accuracy_score, roc_curve, auc

# -------------------------------
# Argument Parser
# -------------------------------

parser = argparse.ArgumentParser(description="Train Montage Classifier.")

parser.add_argument("--architecture", type=str, default= "vit_small_patch14_dinov2.lvd142m",
                    help="Model architecture (e.g. resnet18, vit_base_patch16_224, efficientnet_b0, vit_small_patch14_dinov2.lvd142m)")
parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes.")
parser.add_argument("--loss_fn", type=str, default="LabelSmoothing", choices=["BCE", "Focal", "LabelSmoothing"], help="Loss function type.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--logwandb", type=lambda x: (str(x).lower() == "true"), default=False,
                    help="Enable logging to Weights & Biases.")
parser.add_argument("--data_dir", type=str, default="../data_processing/",
                    help="Directory containing fold csvs and images/")
parser.add_argument("--image_dir_name", type=str, default="images",
                    help="Subfolder name where montage images are stored")
parser.add_argument("--fold", type=int, default=5, help="Fold number.")
parser.add_argument("--parthreshold", type=int, default=10, help="Threshold for binarizing PAR scores")

args = parser.parse_args()

# -------------------------------
# Configuration
# -------------------------------
architecture = args.architecture
num_classes = args.num_classes
loss_fn = args.loss_fn
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
fold = args.fold
logwandb = args.logwandb
par_threshold = args.parthreshold

# Define device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
train_csv = os.path.join(args.data_dir, f"fold{fold}_train.csv")
val_csv = os.path.join(args.data_dir, f"fold{fold}_val.csv")
test_csv = os.path.join(args.data_dir, f"fold{fold}_test.csv")
final_test_csv = os.path.join(args.data_dir, f"final_test.csv")
image_dir = os.path.join(args.data_dir, args.image_dir_name)

# Verification
for p in [train_csv, val_csv, test_csv, final_test_csv, image_dir]:
    if not os.path.exists(p):
        print(f"[ERROR] Path not found: {p}")
        sys.exit(1)

# Input Size Determination
if "vit" in architecture or "dinov2" in architecture:
    input_size = (518, 518) 
    if "384" in architecture: input_size = (384, 384)
else:
    input_size = (224, 224)

print(f"[INFO] Input Resolution: {input_size}")

if logwandb:
    wandb.init(
        project="Malocclusion-classification-MICCAI",
        name=f"MONTAGE_fold{fold}_{architecture}_{loss_fn}",
        config=args
    )

# -------------------------------
# Dataset Class
# -------------------------------
class MontageDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        visit_name = os.path.splitext(row["visit"])[0]
        visit_par = int((row["PAR"]))
        
        # Assumption: Montage images are saved as {visit_name}.png
        img_name = visit_name + ".png" 
        img_path = os.path.join(self.image_dir, img_name)
        
        # Fallback if png not found, try jpg
        if not os.path.exists(img_path):
             img_path = os.path.join(self.image_dir, visit_name + ".jpg")

        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[WARNING] Image not found: {img_path}. returning black image.")
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)

        # Label Processing
        class_v = int(row["class"])
        if num_classes == 2:
            if par_threshold == 10:
                class_v = int(class_v/2) 
            elif par_threshold == 5:
                class_v = int(class_v!=0) 

        return img, torch.tensor(class_v, dtype=torch.long), visit_name, visit_par

# -------------------------------
# Transforms & Helper Functions
# -------------------------------
def pad_to_square(image, fill=0):
    w, h = image.size
    max_wh = max(w, h)
    padding = ((max_wh - w) // 2, (max_wh - h) // 2, (max_wh - w + 1) // 2, (max_wh - h + 1) // 2)
    return F.pad(image, padding, fill=fill, padding_mode='constant')

# Transforms
train_transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize(input_size),
    transforms.RandomResizedCrop(input_size[0], scale=(0.90, 1.0), ratio=(1.0, 1.0)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9,1.1)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def plot_roc_curve_multiclass(all_labels, all_probs, num_classes, save_path):
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        #plt.figure(figsize=(6, 6))
        #plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
        #plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        #plt.legend()
        #plt.title('Montage ROC')
        #plt.grid(True)
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.close()

# -------------------------------
#  Data Loaders
# -------------------------------
train_df = pd.read_csv(train_csv, sep=",")
val_df = pd.read_csv(val_csv, sep=",")
test_df = pd.read_csv(test_csv, sep=",")
final_test_df = pd.read_csv(final_test_csv, sep=",")

train_dataset = MontageDataset(train_df, image_dir, transform=train_transform)
val_dataset = MontageDataset(val_df, image_dir, transform=val_transform)
test_dataset = MontageDataset(test_df, image_dir, transform=val_transform)
final_test_dataset = MontageDataset(final_test_df, image_dir, transform=val_transform)

# Weighted Sampler
class_counts = train_df['class'].value_counts().to_dict()
weights = [1.0 / class_counts.get(x, 1) for x in train_df['class']]
sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32), num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
final_test_loader = DataLoader(final_test_dataset, batch_size=1, shuffle=False)

# -------------------------------
# Model Setup
# -------------------------------
print(f"[INFO] Creating model: {architecture}")
model = timm.create_model(
    architecture,
    pretrained=True,
    num_classes=num_classes,
    drop_rate=0.2
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

if loss_fn == "LabelSmoothing":
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
elif loss_fn == "Focal":
    sys.path.append("./Focal-loss-PyTorch") # Ensure path exists or remove
    try:
        from focal_loss import FocalLoss
        criterion = FocalLoss(gamma=2, alpha=[1.0]*num_classes, task_type="multi-class", num_classes=num_classes)
    except ImportError:
        print("[WARN] FocalLoss not found, using CE.")
        criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss()

# -------------------------------
#  Training Loop
# -------------------------------
best_f1 = 0.0
epochs_no_improve = 0
patience = 8

save_dir = f"./classification_montage/result/Montage_fold{fold}_{architecture}_{loss_fn}_{lr}"
os.makedirs(save_dir, exist_ok=True)
best_model_path = os.path.join(save_dir, "best_model.pth")

print("\n[INFO] Starting Training...")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for inputs, labels, _, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, _, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    
    # Metrics
    if len(all_labels) > 0:
        weighted_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
    else:
        weighted_f1 = 0.0
        acc = 0.0
        
    # Save Best
    if weighted_f1 > best_f1:
        epochs_no_improve = 0
        best_f1 = weighted_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model epoch {epoch+1} | F1: {best_f1:.4f}")
    else:
        epochs_no_improve += 1
        
    if logwandb:
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "val_f1": weighted_f1, "val_acc": acc})
        
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val F1: {weighted_f1:.4f}")
    
    if epochs_no_improve >= patience:
        print("Early Stopping.")
        break

# -------------------------------
# Evaluation Routine (Fold Test & Final Test)
# -------------------------------
def run_evaluation(loader, mode_name="Test"):
    print(f"\nEvaluating on {mode_name}...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_visits = []
    all_pars = []
    
    with torch.no_grad():
        for inputs, labels, visits, pars in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_visits.extend(visits)
            all_pars.extend(pars)
            
    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"{mode_name} Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save Report
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    prefix = "final_" if "Final" in mode_name else ""
    df_report.to_csv(os.path.join(save_dir, f"{prefix}performance_metrics.csv"))
    
    # Save Detailed Predictions
    df_detailed = pd.DataFrame({
        "visit_name": all_visits,
        "PAR": [int(p) for p in all_pars],
        "true_label": all_labels,
        "predicted_label": all_preds
    })
    
    # Add probs
    probs_np = np.array(all_probs)
    for i in range(probs_np.shape[1]):
        df_detailed[f"prob_class_{i}"] = probs_np[:, i]
        
    df_detailed.to_csv(os.path.join(save_dir, f"{prefix}detailed_predictions.csv"), index=False)
    
    # Plot ROC
    roc_path = os.path.join(save_dir, f"{prefix}roc_curve.png")
    plot_roc_curve_multiclass(all_labels, all_probs, num_classes, roc_path)
    
    return f1, acc

# Run Evaluations
test_f1, test_acc = run_evaluation(test_loader, mode_name="Fold Test")
final_f1, final_acc = run_evaluation(final_test_loader, mode_name="Final Test")

if logwandb:
    wandb.log({
        "test_f1": test_f1, 
        "test_acc": test_acc,
        "final_test_f1": final_f1,
        "final_test_acc": final_acc
    })
    wandb.finish()
