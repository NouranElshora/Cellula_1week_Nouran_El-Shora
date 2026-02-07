# LSTM Toxic Text Classifier

## Overview
This folder contains the implementation of an **LSTM-based text classification model** trained on a toxic text dataset.  
The goal is to classify text queries into **9 categories** using deep learning and report evaluation metrics including **F1 score**, confusion matrix, and training curves.

---

## Project Structure
LSTM/
├── toxic_text.ipynb # Main Jupyter notebook with code
├── cellula toxic data (1).csv # Dataset
├── LSTM_results.pdf # PDF with metrics, confusion matrix, and training curves
└── lstm_toxic_classifier.keras # Trained LSTM model

---

## Methodology

1. **Data Preprocessing**
   - Convert text to lowercase and strip whitespace.
   - Encode labels using `LabelEncoder`.
   - Split dataset into **train (80%)** and **test (20%)** sets.
   - Handle class imbalance using **class weights**.

2. **Tokenization & Padding**
   - Tokenized text with a vocabulary size of **20,000**.
   - Padded sequences to a maximum length of **100 tokens**.

3. **LSTM Model Architecture**
   - Embedding layer (128 dimensions)  
   - SpatialDropout1D (0.3)  
   - Bidirectional LSTM (64 units, 0.3 dropout)  
   - Dense layer (64 units, ReLU)  
   - Output layer (9 units, softmax)

4. **Training**
   - Loss: `sparse_categorical_crossentropy`  
   - Optimizer: `Adam`  
   - Metrics: `accuracy`  
   - Early stopping applied with `patience=3` to avoid overfitting.

---

## Evaluation Results

**Macro F1 Score:** 0.7384  
**Weighted F1 Score:** 0.6289  

### Sample Classification Report

| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: |
| **Child Sexual Exploitation** | 1.00 | 1.00 | 1.00 | 21 |
| **Elections** | 1.00 | 1.00 | 1.00 | 22 |
| **Non-violent Crimes** | 0.43 | 0.35 | 0.39 | 60 |
| **Safe** | 0.74 | 0.70 | 0.72 | 199 |
| **Sex-related Crimes** | 0.96 | 1.00 | 0.98 | 23 |
| **Suicide & Self-harm** | 0.96 | 1.00 | 0.98 | 23 |
| **Unknown S-type** | 0.64 | 0.46 | 0.54 | 39 |
| **Unsafe** | 0.50 | 0.80 | 0.62 | 55 |
| **Violent Crimes** | 0.43 | 0.42 | 0.43 | 158 |
| --- | --- | --- | --- | --- |
| **Macro F1 Score** | | | **0.7384** | **600** |
| **Weighted F1 Score** | | | **0.6289** | **600** |

**Confusion Matrix, Loss, and Accuracy Curves** are saved in `LSTM_results.pdf`.

---

