#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study: Early Fusion vs Late Fusion
"""

import os
import sys
import timm
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, roc_curve, auc

# -------------------------------
# Argument Parser
# -------------------------------

parser = argparse.ArgumentParser(description="Train image classifier with MIL, Early, or Late Fusion.")

# FUSION STRATEGY ARGUMENT
parser.add_argument("--fusion", type=str, default="mil", choices=["early", "late"],
                    help="Fusion strategy: 'early' (Concat Features), 'late' (Take Max Treatment need Logits)")

parser.add_argument("--architecture", type=str, default="vit_small_patch14_dinov2.lvd142m", #efficientnet_b0,vit_small_patch14_dinov2.lvd142m
                    help="Model backbone.")
parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes.")
parser.add_argument("--loss_fn", type=str, default="LabelSmoothing", choices=["BCE", "Focal", "LabelSmoothing"], 
                    help="Loss function type.")
parser.add_argument("--lr", type=float, default=0.001, help="Initial Learning rate.")
parser.add_argument("--fold", type=int, default=1, help="Fold number.")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--logwandb", type=lambda x: (str(x).lower() == "true"), default=False,
                    help="Enable logging to Weights & Biases.")
parser.add_argument("--data_dir", type=str, default="./dataset/",
                    help="Directory containing csvs and images/")
parser.add_argument("--drop", type=float, default=0.1, help="Dropout rate.")
parser.add_argument("--usecrop", type=lambda x: (str(x).lower() == "true"), default=True,
                    help="Use smart cropped images.")
parser.add_argument("--sharedbb", type=lambda x: (str(x).lower() == "true"), default=True,
                    help="Use shared backbone.") 
parser.add_argument("--parthreshold", type=int, default=10, help="Threshold for binarizing PAR scores")

args = parser.parse_args()

# -------------------------------
# Setup & Configuration
# -------------------------------
fusion_mode = args.fusion # <--- NEW
architecture = args.architecture
num_classes = args.num_classes
loss_fn = args.loss_fn
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
logwandb = args.logwandb
shared_backbone = args.sharedbb
drop_rate = args.drop
fold = args.fold
par_threshold = args.parthreshold
dtype = "crop" if args.usecrop else "full"

patience = 12 
print(f"[INFO] Running Fusion Mode: {fusion_mode.upper()}")
print(f"[INFO] Network's patience is {patience} epochs.")

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine Image Directory
train_csv = os.path.join(args.data_dir, f"fold{fold}_train.csv")
val_csv = os.path.join(args.data_dir, f"fold{fold}_val.csv")
test_csv = os.path.join(args.data_dir, f"fold{fold}_test.csv")
final_test_csv = os.path.join(args.data_dir, f"final_test.csv")

if args.usecrop:
    image_dir = os.path.join(args.data_dir, "smart_crop")
else:
    image_dir = os.path.join(args.data_dir, "all")

# Check paths
for p in [train_csv, val_csv, test_csv, final_test_csv, image_dir]:
    if not os.path.exists(p):
        print(f"[ERROR] Path not found: {p}")
        sys.exit(1)

if logwandb:
    wandb.init(
        project="Malocclusion-classification-MICCAI",
        name=f"{fusion_mode.upper()}_fold{fold}_{dtype}_{architecture}",
        config={
            "architecture": architecture,
            "fusion": fusion_mode,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "fold": fold,
            "learning_rate": lr
        }
    )

# -------------------------------
#  Flexible Model Architecture ( Early / Late)
# -------------------------------

class AttentionMIL(nn.Module):
    """Attention Mechanism for Multi-Instance Learning."""
    def __init__(self, feat_dim):
        super(AttentionMIL, self).__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(128, 1),
            #nn.Sigmoid() 
        )

    def forward(self, H): # H: (B, N_views, feat_dim)
        A_V = self.attention_V(H) 
        A_U = self.attention_U(A_V) 
        A = torch.softmax(A_U, dim=1) 
        Z = torch.sum(A * H, dim=1) 
        return Z, A

class UniversalMultiViewClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=2, pretrained=True, fusion_mode='mil', drop_rate=0.1):
        super(UniversalMultiViewClassifier, self).__init__()
        self.fusion_mode = fusion_mode
        
        # --- BACKBONE ---
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0, 
            dynamic_img_size=True 
        )
        feat_dim = self.backbone.num_features
        
        # --- FUSION LOGIC ---
        if self.fusion_mode == 'mil':
            print(f"[Model] Initializing Attention MIL")
            self.aggregator = AttentionMIL(feat_dim)
            input_dim = feat_dim
            
        elif self.fusion_mode == 'early':
            print(f"[Model] Initializing Early Fusion (Concat)")
            # In early fusion, we concat 5 vectors of size feat_dim
            input_dim = feat_dim * 5
            self.aggregator = None
            
        elif self.fusion_mode == 'late':
            print(f"[Model] Initializing Late Fusion (Ensemble)")
            # In late fusion, classifier runs on individual images
            input_dim = feat_dim
            self.aggregator = None
            
        # --- CLASSIFIER HEAD ---
        # We maintain the exact same head structure for fair comparison
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim), 
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate), 
            nn.Linear(256, num_classes)
        )

    def forward(self, right, front, left, up, down):
        all_views = [right, front, left, up, down]
        
        # 1. Extract features for all views
        # Result: List of tensors, each (B, feat_dim)
        features = [self.backbone(v) for v in all_views]
        
        # --- MIL PATH ---
        if self.fusion_mode == 'mil':
            # Stack: (B, 5, feat_dim)
            H = torch.stack(features, dim=1) 
            Z, A = self.aggregator(H)
            return self.classifier(Z) # Z is (B, feat_dim)

        # --- EARLY FUSION PATH (Concatenation) ---
        elif self.fusion_mode == 'early':
            # Concat: (B, 5 * feat_dim)
            concat_features = torch.cat(features, dim=1)
            return self.classifier(concat_features)

        # --- LATE FUSION PATH (Decision Averaging) ---
        elif self.fusion_mode == 'late':
            # Run classifier on EACH view independently
            # logits shape: List of 5 tensors, each (B, num_classes)
            view_logits = [self.classifier(f) for f in features]
            
            # Stack: (B, 5, num_classes)
            stacked_logits = torch.stack(view_logits, dim=1)
            
            # Take max of the logits so that if a view finds abnormal it can shout it
            # max_logits = torch.max(stacked_logits, dim=1)
            # ---------------------------------------------------------
            # INCORRECT: avg_logits = torch.mean(stacked_logits, dim=1)
            # RISKY:     max_logits, _ = torch.max(stacked_logits, dim=1)
            # ---------------------------------------------------------
            
            # CORRECT APPROACH:
            # 1. Look at the "Abnormal" column (index 1) for all views
            abnormal_scores = stacked_logits[:, :, 1] # Shape (B, 5)
            
            # 2. Find the index of the view with the highest abnormality score
            # indices shape: (B,)
            best_view_indices = torch.argmax(abnormal_scores, dim=1)
            
            # 3. Select the full logits [Normal, Abnormal] from that winning view
            # We use gather or simple indexing
            # Result shape: (B, 2)
            batch_indices = torch.arange(stacked_logits.size(0), device=stacked_logits.device)
            selected_logits = stacked_logits[batch_indices, best_view_indices]
            
            return selected_logits

# -------------------------------
# Custom Dataset
# -------------------------------
class OrthodonticDataset(torch.utils.data.Dataset):
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
        
        # Construct filenames
        suffixes = ["_left.png", "_right.png", "_center.png", "_up.png", "_down.png"]
        # Note: Order here must match the order in forward() call: right, front, left, up, down
        # Adjusting to match your forward signature: right, center, left, up, down
        
        img_right = Image.open(os.path.join(self.image_dir, visit_name + "_right.png")).convert("RGB")
        img_center = Image.open(os.path.join(self.image_dir, visit_name + "_center.png")).convert("RGB")
        img_left = Image.open(os.path.join(self.image_dir, visit_name + "_left.png")).convert("RGB")
        img_up = Image.open(os.path.join(self.image_dir, visit_name + "_up.png")).convert("RGB")
        img_down = Image.open(os.path.join(self.image_dir, visit_name + "_down.png")).convert("RGB")

        if self.transform:
            img_right = self.transform(img_right)
            img_center = self.transform(img_center)
            img_left = self.transform(img_left)
            img_up = self.transform(img_up)
            img_down = self.transform(img_down)

        class_v = int(row["class"])
        if(num_classes == 2): 
            if(par_threshold == 10):
                class_v = int(class_v/2) 
            elif(par_threshold == 5):
                class_v = int(class_v!=0) 
    
        return img_right, img_center, img_left, img_up, img_down, torch.tensor(class_v, dtype=torch.long), visit_name, visit_par

# -------------------------------
# Utility Functions
# -------------------------------
def plot_roc_curve_multiclass(all_labels, all_probs, num_classes, save_path):
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        #plt.figure(figsize=(6, 6))
        #plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
        #plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title(f'ROC Curve ({fusion_mode.upper()})')
        #plt.legend()
        #plt.grid(True)
    else:
        # Multi-class implementation placeholder
        pass

    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.close()
    
def pad_to_square(image, fill=0):
    w, h = image.size
    max_wh = max(w, h)
    padding = ((max_wh - w) // 2, (max_wh - h) // 2, (max_wh - w + 1) // 2, (max_wh - h + 1) // 2)
    return F.pad(image, padding, fill=fill, padding_mode='constant')

class RandomGaussianBlur:
    def __init__(self, kernel_sizes=(3,5,7), sigma=(0.2, 2.0), p=0.5):
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.p = p
    def __call__(self, img):
        if np.random.rand() < self.p:
            k = np.random.choice(self.kernel_sizes)
            sigma = np.random.uniform(self.sigma[0], self.sigma[1])
            return transforms.functional.gaussian_blur(img, kernel_size=(k,k), sigma=sigma)
        return img

# -------------------------------
# Transforms
# -------------------------------
if "dinov2" in architecture:
    input_size = (518, 518) 
else:
    input_size = (224, 224) 

train_transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize(input_size),
    transforms.RandomResizedCrop(input_size[0], scale=(0.90, 1.0), ratio=(1.0, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), shear=2), 
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    RandomGaussianBlur(kernel_sizes=(3,5), sigma=(0.1, 1.0), p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# -------------------------------
#  Model Initialization
# -------------------------------
model = UniversalMultiViewClassifier(
    backbone_name=architecture,
    num_classes=num_classes,
    pretrained=True,
    fusion_mode=fusion_mode, # Pass the mode here
    drop_rate=drop_rate
).to(device)

# Load CSVs
train_df = pd.read_csv(train_csv, sep=",")
val_df = pd.read_csv(val_csv, sep=",")
test_df = pd.read_csv(test_csv, sep=",")
final_test_df = pd.read_csv(final_test_csv, sep=",")

# Datasets
train_dataset = OrthodonticDataset(train_df, image_dir, transform=train_transform)
val_dataset = OrthodonticDataset(val_df, image_dir, transform=val_transform)
test_dataset = OrthodonticDataset(test_df, image_dir, transform=val_transform)
final_test_dataset = OrthodonticDataset(final_test_df, image_dir, transform=val_transform)

# Sampler
class_counts = train_df['class'].value_counts().to_dict()
# Handle potential missing keys in class_counts if split is small
weights = []
for x in train_df['class']:
    count = class_counts.get(x, 1) # Default to 1 to avoid div by zero
    weights.append(1.0 / count)
samples_weight = torch.tensor(weights, dtype=torch.float32)
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

# Loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
final_test_loader = DataLoader(final_test_dataset, batch_size=1, shuffle=False)

# --- FREEZING LOGIC UPDATED ---
def freeze_backbones_permanently(model):
    # Freezing backbone but allowing head training
    for name, param in model.named_parameters():
        # Keep gradients for MIL attention, Classifier, or Aggregators
        if 'aggregator' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Backbone Frozen. Training ONLY Head parameters: {num_trainable}")

freeze_backbones_permanently(model)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                        lr=lr, 
                        weight_decay=0.02) 

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

if loss_fn == "LabelSmoothing":
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
elif loss_fn == "Focal":
    sys.path.append("./Focal-loss-PyTorch")
    from focal_loss import FocalLoss
    alpha = [1.0] * num_classes
    criterion = FocalLoss(gamma=2, alpha=alpha, task_type="multi-class", num_classes=num_classes)
else:
    criterion = nn.CrossEntropyLoss()

# -------------------------------
# Training Loop
# -------------------------------
best_f1 = 0.0
epochs_no_improve = 0
mod = 'binary' if num_classes == 2 else 'multi'

# Update Save Directory to include Fusion Mode
save_dir = f"./classification_montage/result/{fusion_mode.upper()}_{dtype}_par{par_threshold}_fold{fold}_{drop_rate}_{mod}_{architecture}_K{num_classes}_{loss_fn}_{lr}_b{batch_size}_sched"
best_model_path = os.path.join(save_dir, "best_model.pth")
os.makedirs(save_dir, exist_ok=True)

print(f"\n[INFO] Starting Training: {fusion_mode.upper()} ...")
    
for epoch in range(num_epochs):
    
    model.train()
    train_loss = 0.0
    
    for right, center, left, up, down, labels, _, _ in train_loader:
        optimizer.zero_grad()
        inputs = [v.to(device) for v in [right, center, left, up, down]]
        labels = labels.to(device)
        
        outputs = model(*inputs)
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
        for right, center, left, up, down, labels, _, _ in val_loader:
            inputs = [v.to(device) for v in [right, center, left, up, down]]
            labels = labels.to(device)
            
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
    avg_val_loss = val_loss / len(val_loader)
    
    # Metrics
    total = len(all_labels)
    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    val_accuracy = 100 * correct / total
    
    if total > 0:
        weighted_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        weighted_f1 = 0.0

    # Save Best Model Logic
    improved = weighted_f1 > best_f1
    if improved:
        epochs_no_improve = 0
        best_f1 = weighted_f1
        torch.save(model.state_dict(), best_model_path)
        if logwandb: wandb.log({"best_val_f1": best_f1})
        print(f"Saved new best model at epoch {epoch+1} (F1 = {best_f1:.4f})")
    else:
        epochs_no_improve += 1
    
    # Logging
    current_lr = optimizer.param_groups[0]['lr']
    if logwandb:
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": weighted_f1,
            "lr": current_lr
        })
    
    print(f"Epoch [{epoch+1}/{num_epochs}] LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val F1: {weighted_f1:.4f} | Val Acc: {val_accuracy:.2f}%")
    
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {patience} epochs.")
        break

# -------------------------------
# Evaluation Function
# -------------------------------
def run_evaluation(loader, mode_name="Test"):
    print(f"\nEvaluating best model on {mode_name} set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_visit_names = []
    all_par_vals = []

    with torch.no_grad():
        for right, center, left, up, down, labels, visit_name, visit_par in loader:
            inputs = [v.to(device) for v in [right, center, left, up, down]]
            labels = labels.to(device)
            outputs = model(*inputs)
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_visit_names.extend(visit_name)
            all_par_vals.append(int(visit_par))

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    print(f"{mode_name} Confusion Matrix:\n", cm)
    print(f"{mode_name} Accuracy: {accuracy:.4f}")
    print(f"{mode_name} F1-score: {f1:.4f}")

    # CSV Saving
    df_report = pd.DataFrame(report).transpose()
    summary = pd.DataFrame({
        "accuracy": [accuracy],
        "precision_macro": [precision],
        "recall_macro": [recall],
        "f1_macro": [f1],
        "support": [len(all_labels)]
    }, index=["macro avg"])

    df_final = pd.concat([df_report, summary], axis=0)
    prefix = "final_" if "Final" in mode_name else ""
    metrics_csv_path = os.path.join(save_dir, f"{prefix}performance_metrics.csv")
    df_final.to_csv(metrics_csv_path, index=True)

    # Detailed Predictions
    df_detailed = pd.DataFrame({
        "visit_name": all_visit_names,
        "PAR": all_par_vals,
        "true_label": all_labels,
        "predicted_label": all_preds
    })
    probs_np = np.array(all_probs)
    for i in range(probs_np.shape[1]):
        df_detailed[f"prob_class_{i}"] = probs_np[:, i]
    
    detailed_csv_path = os.path.join(save_dir, f"{prefix}detailed_predictions.csv")
    df_detailed.to_csv(detailed_csv_path, index=False)

    # ROC
    roc_path = os.path.join(save_dir, f"{prefix}roc_curve.png")
    plot_roc_curve_multiclass(all_labels, all_probs, num_classes, roc_path)
    
    return f1, accuracy

# Run on Fold Test
test_f1, test_acc = run_evaluation(test_loader, mode_name="Fold Test")

if logwandb:
    wandb.log({"test_f1": test_f1, "test_accuracy": test_acc})

# Run on Final Test
final_f1, final_acc = run_evaluation(final_test_loader, mode_name="Final Test")

if logwandb:
    wandb.log({"final_test_f1": final_f1, "final_test_accuracy": final_acc})
    wandb.finish()
