# Heart Activity Classification (Time-Series Signals)

## Project
**Heart Activity Classification using Interval-Based Time-Series Modeling**

---

## Description
This project predicts **heart activity classes (0–4)** from multivariate sensor time-series data.  
The goal is to classify each signal sequence into one of five categories using an **interval-based ensemble classifier**.  
The project demonstrates preprocessing, feature engineering, model training, and prediction generation for unseen test data.

---

## Dataset
- **Training data:** `train.csv` (features + labels)  
- **Test data:** `test.csv` (features only)  
- Each row represents a **time-series sample**, with multiple signal readings as features.

---

## Methodology
1. **Data Preprocessing**  
   - Convert all values to numeric, fill missing values with 0  
   - Scale features using Min-Max normalization  
   - Convert 2D data into **nested time-series format** required for interval-based modeling  

2. **Modeling**  
   - Train a **TimeSeriesForestClassifier** (sktime) with 200 estimators  
   - Fit the model on training data and predict labels for test data  

3. **Evaluation**  
   - Predictions are saved as a CSV with index and predicted class columns  
   - Model performance evaluated using standard classification metrics (accuracy, confusion matrix) if true labels are available  

---

## Results
- Successfully generated predictions for test dataset  


