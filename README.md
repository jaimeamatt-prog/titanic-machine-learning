# Titanic Survival Prediction — Machine Learning

## Project Overview

This project explores the Titanic dataset from Kaggle to predict passenger survival using supervised Machine Learning techniques.

The objective is to compare different classification algorithms, evaluate their performance, optimize hyperparameters and select a model based on cross-validation.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow and Keras
- Jupyter Notebook

## Project Workflow

**1. Data Preprocessing**

Data cleaning, feature engineering, categorical encoding and feature scaling.

**2. Model Training**

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machines (SVM)
- Neural Network (scikit-learn)
- Neural Network (Keras)

**3. Model Evaluation**

Models were compared using accuracy, precision, recall, F1-score, ROC-AUC and 10-fold cross-validation.

Hyperparameter optimization and an additional experiment using 30 different train/test splits were performed to evaluate model stability.

## Results

| Metric | Result |
|---|---|
| Best model by cross-validation | Optimized Random Forest |
| Cross-validation accuracy | 83.57% |
| Local test accuracy (Random Forest) | 79.33% |
| Best local test accuracy (Optim