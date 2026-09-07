# Capstone Project 1: Working with NumPy Matrices (Multidimensional Data)

## Project Overview
This project completes **Capstone Project 1: Working with NumPy Matrices (Multidimensional Data)** based on the **National Health and Nutrition Examination Survey (NHANES)** dataset (adult male and female body measurements).

The primary deliverable is [`capstone_project_1.ipynb`](file:///c:/Users/sudee/OneDrive/Documents/New%20projects/Data%20science%20project%201/capstone_project_1.ipynb), a fully documented and pre-executed Jupyter Notebook structured as a formal analytics report. It demonstrates high-performance, multidimensional array computing in **NumPy**, statistical visualization in **Matplotlib**, and rigorous statistical inference using **SciPy**.

---

## Dataset Description
Extracted from the CDC NHANES pre-pandemic examination cycle (`P_BMX` and `P_DEMO`):
- **`nhanes_adult_male_bmx_2020.csv`**: Body measurements of 4,081 adult males ($\ge 18$ years old).
- **`nhanes_adult_female_bmx_2020.csv`**: Body measurements of 4,221 adult females ($\ge 18$ years old).

### Initial 7 Attributes:
1. `weight` ($kg$)
2. `standing height` ($cm$)
3. `upper arm length` ($cm$)
4. `upper leg length` ($cm$)
5. `arm circumference` ($cm$)
6. `hip circumference` ($cm$)
7. `waist circumference` ($cm$)

---

## Key Project Tasks & Methodology

1. **Automated Data Acquisition**: Downloaded directly from the source repository and checked for integrity.
2. **NumPy Matrix Ingestion**: Parsed into 2D floating-point matrices `male` $(4081 \times 7)$ and `female` $(4221 \times 7)$, with zero missing values.
3. **Comparative Weight Histograms**: Stacked subplots (`plt.subplot(2, 1, ·)`) with strictly synchronized x-axis boundaries (`plt.xlim`) and visual markers for means and medians.
4. **Side-by-Side Boxplots**: Non-parametric five-number comparison of male vs. female weights highlighting sexual dimorphism and upper-tail outliers.
5. **Numerical Aggregation**: Complete evaluation of central tendency (mean, median, $Q_1$, $Q_3$), dispersion (variance, std, IQR, range), and distribution shape (Fisher-Pearson skewness and excess kurtosis).
6. **BMI Feature Engineering**: Calculated $\text{BMI} = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2}$ and appended as column 8 to the `female` matrix.
7. **$Z$-Score Standardization**: Built `zfemale` where each feature is standardized to mean $0.0$ and unit variance $1.0$.
8. **Multivariate Correlation & Pairplot**: High-resolution $5 \times 5$ scatterplot matrix with diagonal histograms for Height, Weight, Waist, Hip, and BMI, accompanied by Pearson ($r$) and Spearman ($\rho$) correlation matrices.
9. **Central Adiposity Ratios**: Calculated Waist-to-Height Ratio (WHtR) and Waist-to-Hip Ratio (WHR), appending them as new columns to both `male` and `female` matrices.
10. **4-Box Comparative Boxplot**: Side-by-side boxplots comparing `[Female WHtR, Male WHtR, Female WHR, Male WHR]` against WHO risk benchmarks.
11. **Analytical Appraisal**: Comprehensive comparative appraisal of BMI vs. WHtR vs. WHR advantages and limitations.
12. **Extreme Phenotype Analysis**: Utilized `np.argsort` to isolate and clinically interpret the 5 lowest and 5 highest BMI female participants.

---

## Deliverables Summary

| File | Purpose |
| :--- | :--- |
| [`capstone_project_1.ipynb`](file:///c:/Users/sudee/OneDrive/Documents/New%20projects/Data%20science%20project%201/capstone_project_1.ipynb) | Master Jupyter Notebook containing all 12 tasks, embedded charts, markdown discussions, and verified outputs. |
| [`create_capstone_notebook.py`](file:///c:/Users/sudee/OneDrive/Documents/New%20projects/Data%20science%20project%201/create_capstone_notebook.py) | Automation script that programmatically builds and executes the notebook. |
| `nhanes_adult_male_bmx_2020.csv` | Raw NHANES adult male dataset. |
| `nhanes_adult_female_bmx_2020.csv` | Raw NHANES adult female dataset. |
| [`requirements.txt`](file:///c:/Users/sudee/OneDrive/Documents/New%20projects/Data%20science%20project%201/requirements.txt) | Environment dependencies. |

---

## How to Run

1. Clone or navigate to the project directory:
   ```bash
   cd "c:/Users/sudee/OneDrive/Documents/New projects/Data science project 1"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch JupyterLab or Notebook:
   ```bash
   jupyter lab capstone_project_1.ipynb
   ```
   *Alternatively, re-run all cells via command line:*
   ```bash
   python create_capstone_notebook.py
   ```
