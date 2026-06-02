# 🎓 Student Placement Prediction Model

**Classification of Data Using AI - Project 2**  
*DecodeLabs Artificial Intelligence Internship*

**Developed by:** Yakubu Labibat

---

## 📋 Overview

This repository contains a comprehensive machine learning pipeline designed to predict student placement status using historical enrollment data. The project demonstrates end-to-end workflow from data preprocessing through model evaluation, implementing industry-standard practices for binary classification tasks.

**Key Objective:** Build and evaluate a predictive model that accurately classifies whether students will be placed or not based on their academic and personal attributes.

---

## ✨ Features

- **🔧 Data Preprocessing**  
  Robust handling of features with automated missing value detection and target variable isolation for clean data pipelines

- **📏 Feature Scaling**  
  StandardScaler normalization ensures stable convergence and optimal performance for linear models

- **🤖 Classification Pipeline**  
  Scikit-learn Logistic Regression implementation for efficient and interpretable binary classification

- **📊 Performance Evaluation**  
  Comprehensive metrics including precision, recall, F1-score, and visual confusion matrix analysis

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data Manipulation** | NumPy, Pandas |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |

---

## 📊 Model Pipeline

The implementation follows a structured five-step approach:

### 1. **Exploratory Assessment**
   - Load and inspect dataset structure
   - Check for missing values and data quality issues
   - Generate descriptive statistics

### 2. **Data Splitting**
   - Isolate target variable (`placement_status`)
   - Separate features from labels
   - Partition into training and test sets

### 3. **Feature Standardization**
   - Fit StandardScaler on training data
   - Transform both training and test features uniformly
   - Ensure consistent feature scaling across datasets

### 4. **Model Training & Inference**
   - Train LogisticRegression classifier on scaled training data
   - Generate predictions on unseen test set
   - Optimize for binary classification performance

### 5. **Evaluation & Visualization**
   - Generate detailed classification metrics report
   - Create annotated confusion matrix heatmap
   - Analyze model strengths and weaknesses

---

## 📈 Results & Evaluation

### Performance Metrics

The model's predictive performance is evaluated using standard classification metrics:

| Metric | Description |
|--------|-------------|
| **Precision** | Proportion of positive predictions that are correct |
| **Recall** | Proportion of actual positives correctly identified |
| **F1-Score** | Harmonic mean of precision and recall |
| **Accuracy** | Overall correctness of predictions |

*See generated classification report for detailed metrics*

---

## 🎯 Confusion Matrix Analysis

The confusion matrix heatmap provides visual insight into model predictions:

```
                    Predicted
                Not Placed | Placed
Actual  Not Placed    TN   |   FP
        Placed        FN   |   TP
```

### Matrix Components:

| Component | Definition |
|-----------|-----------|
| **True Negatives (TN)** | ✓ Students correctly predicted as not placed |
| **False Positives (FP)** | ✗ Students incorrectly predicted as placed |
| **False Negatives (FN)** | ✗ Students incorrectly predicted as not placed |
| **True Positives (TP)** | ✓ Students correctly predicted as placed |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Usage
```python
# Load and preprocess data
# Train model with standardized features
# Generate predictions and evaluation metrics
# Visualize results with confusion matrix
```

---

## 📁 Project Structure

```
Project 2:classification of data using AI/
├── ReadME.md (this file)
├── student_placement_model.py
├── data/
│   └── placement_data.csv
└── outputs/
    ├── classification_report.txt
    └── confusion_matrix.png
```

---

## 💡 Key Insights

- The model uses standardized features for improved numerical stability
- Logistic Regression provides interpretable probability estimates
- Confusion matrix analysis reveals specific prediction patterns
- Feature scaling is critical for linear model convergence

---

## 📝 Notes

This project serves as a practical demonstration of ML fundamentals, suitable for students learning classification, feature engineering, and model evaluation techniques.

---

**Last Updated:** June 2026  
**Status:** ✅ Complete
