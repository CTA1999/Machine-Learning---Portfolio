# Classification of Handwritten Digits using SVD

## Problem Statement
Classify handwritten digits (0-9) using **Singular Value Decomposition (SVD)**.  
Application: automatic reading of zip codes.

## Dataset
- Training images: `trainInput.csv` (256x1707)
- Training labels: `trainOutput.csv` (1x1707)
- Test images: `testInput.csv` (256x2007)
- Test labels: `testOutput.csv` (1x2007)
- Each image is 16x16 grayscale, reshaped into 256x1 vector.

## Methodology
1. Form a matrix `A` for each digit (rows = images).
2. Perform **SVD** for each `A`. Right singular vectors = “singular images”.
3. Represent test images using the first **k=20 singular images** of each digit.
4. Compute residuals and classify based on **smallest residual**.
5. Evaluate overall and per-digit accuracy using a confusion matrix.

## Results
- Overall classification accuracy: 93.97%
- Confusion matrix: 

## Skills Demonstrated
- Python (NumPy, Pandas, Matplotlib)
- Linear Algebra (SVD)
- Data Analysis & Visualization
- Pattern Recognition
- Model Evaluation (Confusion Matrix)

## How to Run
1. Place the dataset in `data/`.
2. Open the notebook `notebooks/svd_classification.ipynb`.
3. Run the cells sequentially to reproduce results.
