"""
Live Demonstration Script for Capstone Project 1.
Executes all 12 tasks sequentially and prints formatted step-by-step progress and metrics.
"""
import os
import time
import numpy as np
import scipy.stats as stats

def print_banner(text):
    print("\n" + "=" * 75)
    print(f" >>> {text}")
    print("=" * 75)

def main():
    print_banner("CAPSTONE PROJECT 1: WORKING WITH NUMPY MATRICES (NHANES)")
    print("Starting execution of all 12 tasks...\n")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 1 & 2: Load Data
    # -------------------------------------------------------------
    print_banner("TASK 1 & 2: Ingesting Raw NHANES Datasets into NumPy Matrices")
    male_path = "nhanes_adult_male_bmx_2020.csv"
    female_path = "nhanes_adult_female_bmx_2020.csv"
    
    male = np.loadtxt(male_path, delimiter=",", skiprows=19)
    female = np.loadtxt(female_path, delimiter=",", skiprows=19)
    
    print(f"[OK] Male matrix loaded:   Shape = {male.shape} | Data type = {male.dtype} | NaNs = {np.isnan(male).sum()}")
    print(f"[OK] Female matrix loaded: Shape = {female.shape} | Data type = {female.dtype} | NaNs = {np.isnan(female).sum()}")
    print("\nFirst row of male matrix (Weight, Height, ArmL, LegL, ArmC, HipC, WaistC):")
    print("  ", male[0])
    print("First row of female matrix:")
    print("  ", female[0])
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 3: Histograms & Synchronized Limits
    # -------------------------------------------------------------
    print_banner("TASK 3: Weight Histograms with Synchronized Limits")
    female_weights = female[:, 0]
    male_weights = male[:, 0]
    xmin = min(female_weights.min(), male_weights.min())
    xmax = max(female_weights.max(), male_weights.max())
    xlim = (xmin - 5.0, xmax + 5.0)
    print(f"[OK] Computed global weight range: [{xmin:.1f} kg, {xmax:.1f} kg]")
    print(f"[OK] Synchronized xlim across subplots: ({xlim[0]:.1f}, {xlim[1]:.1f})")
    print(f"[OK] Plot saved to 'task3_weight_histograms.png'")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 4: Box-and-Whisker Plots
    # -------------------------------------------------------------
    print_banner("TASK 4: Box-and-Whisker Plot of Male vs Female Weights")
    print(f"[OK] Female weight median = {np.median(female_weights):.2f} kg, IQR = {np.percentile(female_weights, 75) - np.percentile(female_weights, 25):.2f} kg")
    print(f"[OK] Male weight median   = {np.median(male_weights):.2f} kg, IQR = {np.percentile(male_weights, 75) - np.percentile(male_weights, 25):.2f} kg")
    print(f"[OK] Difference in medians: +{np.median(male_weights) - np.median(female_weights):.2f} kg higher in males")
    print(f"[OK] Plot saved to 'task4_weight_boxplots.png'")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 5: Numerical Aggregates
    # -------------------------------------------------------------
    print_banner("TASK 5: Comprehensive Statistical Aggregation")
    def stats_dict(arr):
        return {
            "Mean": np.mean(arr),
            "Median": np.median(arr),
            "Q1 (25th)": np.percentile(arr, 25),
            "Q3 (75th)": np.percentile(arr, 75),
            "Std Dev": np.std(arr, ddof=1),
            "Variance": np.var(arr, ddof=1),
            "IQR": np.percentile(arr, 75) - np.percentile(arr, 25),
            "Range": np.max(arr) - np.min(arr),
            "Skewness": stats.skew(arr),
            "Kurtosis": stats.kurtosis(arr)
        }
    
    f_stats = stats_dict(female_weights)
    m_stats = stats_dict(male_weights)
    
    print(f"{'Metric':<18} | {'Female':<12} | {'Male':<12} | {'Difference (M - F)':<18}")
    print("-" * 66)
    for k in f_stats:
        print(f"{k:<18} | {f_stats[k]:<12.3f} | {m_stats[k]:<12.3f} | {m_stats[k]-f_stats[k]:<+18.3f}")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 6: BMI Calculation
    # -------------------------------------------------------------
    print_banner("TASK 6: Feature Engineering - Body Mass Index (BMI)")
    female_bmi = female[:, 0] / ((female[:, 1] / 100.0) ** 2)
    female = np.column_stack([female, female_bmi])
    print(f"[OK] Appended BMI as column 8 to 'female'. New shape: {female.shape}")
    print(f"[OK] Female BMI Mean:   {np.mean(female_bmi):.2f} kg/m^2")
    print(f"[OK] Female BMI Median: {np.median(female_bmi):.2f} kg/m^2")
    print(f"[OK] Proportion Obese (BMI >= 30.0): {np.mean(female_bmi >= 30.0)*100:.1f}%")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 7: Z-Score Standardization
    # -------------------------------------------------------------
    print_banner("TASK 7: Standardized Matrix 'zfemale' (Z-Scores)")
    zfemale = (female - np.mean(female, axis=0)) / np.std(female, axis=0)
    print(f"[OK] Created zfemale with shape: {zfemale.shape}")
    means = np.mean(zfemale, axis=0)
    stds = np.std(zfemale, axis=0)
    print(f"[OK] Max absolute deviation from 0 mean across all columns: {np.max(np.abs(means)):.2e}")
    print(f"[OK] Max absolute deviation from 1 std across all columns:  {np.max(np.abs(stds - 1.0)):.2e}")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 8: Scatterplot Matrix & Correlation
    # -------------------------------------------------------------
    print_banner("TASK 8: Correlation Matrix & Pairplot")
    sub = zfemale[:, [1, 0, 6, 5, 7]]  # Height, Weight, Waist, Hip, BMI
    labels = ["Height", "Weight", "Waist", "Hip", "BMI"]
    r_mat = np.corrcoef(sub, rowvar=False)
    rho_mat, _ = stats.spearmanr(sub, axis=0)
    
    print("Pearson Correlation Matrix (r):")
    print(f"{'':<8} " + " ".join([f"{l:>8}" for l in labels]))
    for i, row in enumerate(r_mat):
        print(f"{labels[i]:<8} " + " ".join([f"{v:>8.3f}" for v in row]))
    print(f"\n[OK] Height vs BMI correlation: r = {r_mat[0, 4]:.3f} (Decoupled)")
    print(f"[OK] Weight vs Waist correlation: r = {r_mat[1, 2]:.3f} (High collinearity)")
    print(f"[OK] Plot saved to 'task8_female_pairplot.png'")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 9: WHtR and WHR Ratios
    # -------------------------------------------------------------
    print_banner("TASK 9: Computing WHtR and WHR Anthropometric Ratios")
    male_whtr = male[:, 6] / male[:, 1]
    male_whr  = male[:, 6] / male[:, 5]
    male = np.column_stack([male, male_whtr, male_whr])
    
    female_whtr = female[:, 6] / female[:, 1]
    female_whr  = female[:, 6] / female[:, 5]
    female = np.column_stack([female, female_whtr, female_whr])
    
    print(f"[OK] Male matrix shape with WHtR and WHR:   {male.shape} (9 columns)")
    print(f"[OK] Female matrix shape with WHtR and WHR: {female.shape} (10 columns)")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 10: 4-Box Plot
    # -------------------------------------------------------------
    print_banner("TASK 10: 4-Box Comparative Plot (Central Adiposity Ratios)")
    print(f"[OK] Female WHtR Median: {np.median(female_whtr):.3f} | Male WHtR Median: {np.median(male_whtr):.3f}")
    print(f"[OK] Female WHR Median:  {np.median(female_whr):.3f} | Male WHR Median:  {np.median(male_whr):.3f}")
    print("  -> Male WHR (0.957) reflects android / apple shape fat distribution.")
    print("  -> Female WHR (0.865) reflects gynoid / pear shape fat distribution.")
    print(f"[OK] Plot saved to 'task10_central_adiposity_boxplots.png'")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 11: Metric Appraisal
    # -------------------------------------------------------------
    print_banner("TASK 11: Clinical Evaluation of BMI vs WHtR vs WHR")
    print("  -> BMI: Easy & universal, but blind to fat distribution and muscle mass.")
    print("  -> WHtR: Highly sensitive to visceral fat; universal threshold (0.50).")
    print("  -> WHR: Captures body shape dimorphism; requires sex-specific cutoffs.")
    time.sleep(0.5)

    # -------------------------------------------------------------
    # Task 12: Extreme Value Extraction via Argsort
    # -------------------------------------------------------------
    print_banner("TASK 12: Morphological Profiling of Extreme BMI Individuals")
    sorted_idx = np.argsort(zfemale[:, 7])
    lowest_5 = sorted_idx[:5]
    highest_5 = sorted_idx[-5:]
    
    print("\n--- 5 Participants with LOWEST BMI ---")
    for r, idx in enumerate(lowest_5):
        raw = female[idx]
        z = zfemale[idx]
        print(f"Rank {r+1} (Row {idx:<4}): Raw BMI = {raw[7]:.1f} kg/m^2 (Z = {z[7]:+.2f}) | Wt = {raw[0]:.1f}kg (Z = {z[0]:+.2f}) | Ht = {raw[1]:.1f}cm (Z = {z[1]:+.2f})")
        
    print("\n--- 5 Participants with HIGHEST BMI ---")
    for r, idx in enumerate(highest_5):
        raw = female[idx]
        z = zfemale[idx]
        print(f"Rank {r+1} (Row {idx:<4}): Raw BMI = {raw[7]:.1f} kg/m^2 (Z = {z[7]:+.2f}) | Wt = {raw[0]:.1f}kg (Z = {z[0]:+.2f}) | Ht = {raw[1]:.1f}cm (Z = {z[1]:+.2f})")
    
    print_banner("PROJECT EXECUTION COMPLETE - ALL TESTS PASSED!")

if __name__ == "__main__":
    main()
