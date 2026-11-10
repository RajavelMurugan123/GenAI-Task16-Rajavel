# Assignment 16 – SVM, Trees, Ensembles and Validation

## Dataset
- Name: Breast Cancer Wisconsin (Diagnostic) Data Set
- Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Required CSV: `data.csv`
- Target column: `diagnosis` (M = malignant, B = benign)
- Features: numerical measurements of cell nuclei.

Download the Kaggle dataset, extract it, and place `data.csv` in this project's `data/` folder. The CSV is not included in this ZIP. Check the dataset's current license/terms before redistribution.

## Project structure
```text
GenAI-Task16-Rajavel/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── data.csv
└── outputs/
```

## Setup (Windows PowerShell)
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run
```powershell
python main.py
```
The script prints dataset details, model accuracies, classification reports, 5-fold CV scores and feature importance. It saves plots and result CSVs in `outputs/`.

## Tasks implemented
1. SVM with linear and RBF kernels. StandardScaler is inside a Pipeline to prevent scaling leakage.
2. Shallow decision tree (`max_depth=2`) versus unrestricted tree; compare train/test accuracy and save a depth-limited tree plot.
3. 60% train / 20% validation / 20% test split. Select `max_depth` on validation accuracy and evaluate the selected model on the held-out test set.
4. Stratified 5-fold cross-validation; print each fold, mean and standard deviation.
5. Train and compare BaggingClassifier and AdaBoostClassifier.
6. Compare single Decision Tree, Bagging and Random Forest; print and plot Random Forest feature importance.

## Concepts
- **SVM:** finds a decision boundary with a large margin. Linear uses a straight boundary; RBF can model nonlinear boundaries.
- **Decision Tree:** learns feature-based split rules. A shallow tree may underfit; a deep tree may overfit.
- **Train set:** fits model parameters.
- **Validation set:** tunes hyperparameters.
- **Test set:** final evaluation; do not use it for tuning.
- **K-Fold CV:** divides data into k folds and evaluates each fold while training on the others.
- **Bagging:** trains models independently on bootstrap samples and aggregates predictions; often reduces variance.
- **Boosting:** trains learners sequentially, focusing more on previous errors.
- **Random Forest:** combines bagged decision trees and random feature subsets.

## Notes
Accuracy depends on the dataset and library versions. Use the values printed by your own run. Do not invent or copy results.
The code uses scikit-learn's current `estimator=` API. If an older version gives an argument error, upgrade scikit-learn:
`python -m pip install --upgrade scikit-learn`

The assignment title mentions unsupervised learning, but the provided tasks and dataset requirement specify supervised classification only. This project follows the six tasks actually listed.

## Submission
- Put `data.csv` in `data/`.
- Run `python main.py` successfully.
- Review `outputs/`.
- Keep all project files inside this one folder.
- Exclude `.venv/` and `__pycache__/` from the ZIP.
- Submit `GenAI-Task16-Rajavel.zip`.
