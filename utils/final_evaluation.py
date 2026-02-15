#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    roc_auc_score, 
    accuracy_score, 
    roc_curve, 
    auc,
    classification_report,
    cohen_kappa_score
)
from sklearn.utils import resample


# --- Configuration ---
base_dir = "./../weights/"

filename = "final_detailed_predictions.csv"
OUTPUT_PREDICTIONS_FILE = "./dataset/ai_ensemble_predictions.csv" # New output file
RATER_A_ORIGINAL_FILE = "./dataset/rater_A.csv" 
par_threshold = 10
weights_folder = "MIL_crop_fold{}_0.05_binary_vit_small_patch14_dinov2.lvd142m_K2_LabelSmoothing_0.001_b32_sched"
# weights_folder = "MIL_crop_par"+str(par_threshold)+"_fold{}_0.05_binary_vit_small_patch14_dinov2.lvd142m_K2_LabelSmoothing_0.001_b32_sched"
# weights_folder = "LATE_crop_par10_fold{}_0.05_binary_vit_small_patch14_dinov2.lvd142m_K2_LabelSmoothing_0.001_b32_sched"
# weights_folder = "LATE_full_par10_fold{}_0.05_binary_vit_small_patch14_dinov2.lvd142m_K2_LabelSmoothing_0.001_b32_sched"
# weights_folder = "Montage_fold{}_efficientnet_b0_LabelSmoothing_0.0001"
# weights_folder = "Montage_fold{}_vit_small_patch14_dinov2.lvd142m_LabelSmoothing_1e-05"
# weights_folder = "MIL_crop_par10_fold{}_0.05_binary_vit_small_patch14_dinov2.lvd142m_K2_LabelSmoothing_0.001_b32_sched"
# LATE_crop_par10_fold5_0.05_binary_vit_small_patch14_dinov2.lvd142m_K2_LabelSmoothing_0.001_b32_sched
# Lists to store data
all_folds_dfs = []

# --- 1. Load Data ---
print("Loading data...")
for fold in range(1, 6):
    folder_name = weights_folder.format(fold)
    file_path = os.path.join(base_dir, folder_name, filename)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['fold'] = fold
        all_folds_dfs.append(df)
        print(f"Loaded Fold {fold}: {len(df)} visits")
    else:
        print(f"Warning: File not found for Fold {fold}")

if not all_folds_dfs:
    raise ValueError("No data loaded.")

raw_df = pd.concat(all_folds_dfs, ignore_index=True)

# --- 2. ENSEMBLE LOGIC: Soft Voting ---
ensemble_df = raw_df.groupby('visit_name').agg({
    'prob_class_1': 'mean',          
    'true_label': 'first',           
    'prob_class_0': 'mean'
}).reset_index()

# Threshold 0.5
ensemble_df['final_predicted_label'] = (ensemble_df['prob_class_1'] >= 0.5).astype(int)

y_true = ensemble_df['true_label'].values
y_pred = ensemble_df['final_predicted_label'].values
y_prob = ensemble_df['prob_class_1'].values

# --- 3. Compute Basic Metrics ---
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary') 
auroc = roc_auc_score(y_true, y_prob)
cm = confusion_matrix(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

# --- 4. BOOTSTRAP 95% CONFIDENCE INTERVAL FOR KAPPA ---
print("\nCalculating 95% Confidence Interval (1000 bootstraps)...")
n_bootstraps = 1000
rng = np.random.RandomState(42) # Fix seed for reproducibility
bootstrapped_kappas = []

for i in range(n_bootstraps):
    # Resample indices with replacement
    indices = rng.randint(0, len(y_pred), len(y_pred))
    
    if len(np.unique(y_true[indices])) < 2:
        # Skip samples that don't have both classes 
        continue
        
    score = cohen_kappa_score(y_true[indices], y_pred[indices])
    bootstrapped_kappas.append(score)

# Calculate percentiles
sorted_kappas = np.sort(bootstrapped_kappas)
lower_ci = sorted_kappas[int(0.025 * len(sorted_kappas))]
upper_ci = sorted_kappas[int(0.975 * len(sorted_kappas))]

print("\n" + "="*40)
print(f"ENSEMBLE RESULTS (Soft Voting)")
print("="*40)
print(f"Accuracy:        {acc:.4f}")
print(f"F1-Score:        {f1:.4f}")
print(f"AUROC:           {auroc:.4f}")
print(f"Cohen's Kappa:   {kappa:.4f} (95% CI: {lower_ci:.4f} - {upper_ci:.4f})") 
print("-" * 20)
print("Confusion Matrix:")
print(cm)
print("-" * 20)

# --- MODIFIED LINE: Added digits=4 ---
print(classification_report(y_true, y_pred, digits=4))
# -------------------------------------
# --- 5. Plot ROC Curves ---
# plt.figure(figsize=(10, 8))

# Plot Individual Folds
for i, df in enumerate(all_folds_dfs):
    fold_num = df['fold'].iloc[0]
    fpr_fold, tpr_fold, _ = roc_curve(df['true_label'], df['prob_class_1'])
    roc_auc_fold = auc(fpr_fold, tpr_fold)
    # plt.plot(fpr_fold, tpr_fold, color='gray', linestyle='--', alpha=0.35, label=f'Fold {fold_num}')

# Plot Ensemble
fpr_ens, tpr_ens, _ = roc_curve(y_true, y_prob)
roc_auc_ens = auc(fpr_ens, tpr_ens)

# plt.plot(fpr_ens, tpr_ens, color='red', linestyle='-', alpha=1.0, linewidth=3.0, 
#              label=f'Ensemble (AUC = {roc_auc_ens:.3f})')

# plt.plot([0, 1], [0, 1], color='navy', linestyle=':')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])

# # --- UPDATED TEXT SIZES ---
# plt.xlabel('False Positive Rate', fontsize=14)
# plt.ylabel('True Positive Rate', fontsize=14)
# plt.title(f'ROC Curve\nKappa: {kappa:.3f} [{lower_ci:.3f}-{upper_ci:.3f}]', fontsize=18, fontweight='bold')
# plt.legend(loc="lower right", fontsize=12)

# # Increase size of axis tick marks (numbers)
# plt.tick_params(axis='both', which='major', labelsize=15)

# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('ensemble_kappa_ci_roc.png', dpi=300)
# print("\nPlot saved to 'ensemble_kappa_ci_roc.png'")

# plt.show() # Disabled plt.show() for standard script execution

# --- 6. SAVE ENSEMBLE RESULTS ---
# Select and rename the relevant columns for the output file
output_df = ensemble_df[['visit_name', 'true_label', 'final_predicted_label', 'prob_class_1']]
output_df.rename(columns={
    'visit_name': 'visit',
    'true_label': 'True_Class_RaterA',
    'final_predicted_label': 'AI_Predicted_Class',
    'prob_class_1': 'AI_Prob_Class_1'
}, inplace=True)

# Save to CSV
output_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
print(f"\n Ensemble predictions saved to '{OUTPUT_PREDICTIONS_FILE}'")

# --- 7. COMPUTE KAPPA AGAINST RATER_A_ORIGINAL_FILE (WITH MAPPING & CLEANING) ---
print("\n" + "="*40)
print("COMPARISON WITH RATER_A ORIGINAL FILE (BINARY MAPPING)")
print("="*40)
