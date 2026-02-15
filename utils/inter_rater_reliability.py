#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, classification_report

# --- Configuration ---

# 1. PRIMARY DATA (For Main Comparison)
# RATER_A_FILE is the RETEST file (used for comparison against B)
RATER_A_FILE = "./../data/rater_A.csv"  # <--- Rater A (Retest)
RATER_B_FILE = "./../data/rater_B.csv"  # <--- Rater Bs

# 2. RELIABILITY DATA (For Test-Retest Check)
# This is the Original file (used only to check self-consistency against Retest)

PAR_THRESHOLD = 10
PAR_COLUMN = 'PAR' 
KEY_COLUMNS = ['patient', 'visit']
OUTPUT_FILE_FULL = "./../data/inter_rater_Retest_vs_B.csv"

def map_par_to_binary(par_value, threshold):
    """Maps PAR value to binary class (0 or 1)."""
    # 0 = Low Risk (<= Threshold), 1 = High Risk (> Threshold)
    return 1 if par_value > threshold else 0

def load_and_preprocess_data(file_path, rater_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    print(f"Loading data for {rater_name} from {file_path}...")
    df = pd.read_csv(file_path)
    # Compute binary class
    df[f'PAR_CLASS_{rater_name}'] = df[PAR_COLUMN].apply(lambda x: map_par_to_binary(x, PAR_THRESHOLD))
    return df

# =============================================================================
# 1. LOAD MAIN DATA (Retest vs B)
# =============================================================================
try:
    # Load Rater A (Retest)
    df_rater_a_retest = load_and_preprocess_data(RATER_A_FILE, 'RaterA_Retest')
    # Load Rater B
    df_rater_b = load_and_preprocess_data(RATER_B_FILE, 'RaterB')
except Exception as e:
    print(e)
    sys.exit()

print("-" * 40)
print(f"Visits Rater A (Retest): {len(df_rater_a_retest)}")
print(f"Visits Rater B: {len(df_rater_b)}")
print("-" * 40)

# =============================================================================
# 2. MAIN COMPARISON: RATER A (RETEST) vs RATER B
# =============================================================================

merged_main = pd.merge(
    df_rater_a_retest, 
    df_rater_b, 
    on=KEY_COLUMNS, 
    how='inner',
    suffixes=('_ARet', '_B')
)

if len(merged_main) == 0:
    print("Error: No common visits found for Main Comparison.")
    sys.exit()

# Define column names based on merge suffixes
par_A = merged_main['PAR_ARet'].values
par_B = merged_main['PAR_B'].values
class_A = merged_main['PAR_CLASS_RaterA_Retest'].values
class_B = merged_main['PAR_CLASS_RaterB'].values

print("\n" + "="*60)
print(" MAIN COMPARISON RESULTS: Rater A (Retest) vs Rater B")
print("="*60)

# Correlation
corr_main, p_main = pearsonr(par_A, par_B)
print(f"Pearson Correlation: {corr_main:.4f} (p={p_main:.4f})")

# Kappa
kappa_main = cohen_kappa_score(class_A, class_B)
print(f"Cohen's Kappa:      {kappa_main:.4f}")

# F1 Score
f1_main = f1_score(class_A, class_B, average='binary')
print(f"F1 Score (Binary):  {f1_main:.4f}")

# --- CONFUSION MATRIX (MAIN) ---
cm_main = confusion_matrix(class_A, class_B)
print("\nConfusion Matrix (Main):")
print("                  Pred (Rater B)")
print("                 [ <=5   >5 ]")
print(f"True (Retest) <=5 [ {cm_main[0][0]:<3}  {cm_main[0][1]:<3} ]")
print(f"             >5  [ {cm_main[1][0]:<3}  {cm_main[1][1]:<3} ]")


# ICC (Retest vs B)
icc_data = merged_main[['PAR_ARet', 'PAR_B']].copy()
icc_data['Subject'] = merged_main.index
icc_long = pd.melt(icc_data, id_vars=['Subject'], value_vars=['PAR_ARet', 'PAR_B'], var_name='Rater', value_name='Score')


# Save Main Comparison
merged_main.to_csv(OUTPUT_FILE_FULL, index=False)
print(f" Main comparison data saved to: {OUTPUT_FILE_FULL}")


# =============================================================================
# 5. PLOTTING
# =============================================================================

# Scatter: Main Comparison (Retest vs B)
plt.figure(figsize=(6, 6))
plt.scatter(par_A, par_B, alpha=0.6, label='Data Points')
plt.plot([0, 20], [0, 20], 'k--', label='Perfect Agreement')
plt.xlabel("Rater A (Retest)")
plt.ylabel("Rater B")
plt.title("Main: Rater A (Retest) vs Rater B")
plt.legend()
plt.grid(True)
plt.savefig("./../data/scatter_main_retest_vs_B.png")
print("\nProcessing Complete.")
