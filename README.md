# Fake News Detection Using Machine Learning & NLP

[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Three.js](https://img.shields.io/badge/3D-Three.js%20WebGL-black.svg?style=flat&logo=three.js&logoColor=white)](https://threejs.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E.svg?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Design System](https://img.shields.io/badge/UI%2FUX-Cyberpunk%20Dark%20Glass-7C5CFC.svg?style=flat)](#)
[![Status](https://img.shields.io/badge/Status-Complete-10B981.svg?style=flat)](#)
[![SRS Version](https://img.shields.io/badge/SRS-v1.0-blue.svg?style=flat)](SPEC.md)

An end-to-end Machine Learning and Natural Language Processing application that analyzes textual news content and classifies it into two categories: **REAL** and **FAKE**. Built strictly adhering to the project's Software Requirements Specification (SRS v1.0).

---

## 📑 Table of Contents
- [1. Overview](#1-overview)
- [2. System Architecture](#2-system-architecture)
- [3. Key Features](#3-key-features)
- [4. Project Directory Structure](#4-project-directory-structure)
- [5. Dataset Details](#5-dataset-details)
- [6. Installation & Setup](#6-installation--setup)
- [7. How to Run the Application](#7-how-to-run-the-application)
- [8. Model Training & Benchmarks](#8-model-training--benchmarks)
- [9. Programmatic Usage](#9-programmatic-usage)
- [10. Limitations & Ethical Notice](#10-limitations--ethical-notice)
- [11. Future Enhancements](#11-future-enhancements)

---

## 1. Overview

The rapid growth of digital media and social platforms has amplified the proliferation of misleading and fabricated news articles. Manually auditing every article is impossible at scale.

**TruthLens** applies Natural Language Processing (NLP) and Machine Learning (ML) techniques to discover linguistic patterns in labeled news datasets, providing:
- Real-time classification for submitted news articles or headlines.
- Calibrated confidence estimation.
- Transparent preprocessed text inspection.
- Diagnostic evaluation metrics (Confusion Matrix, Precision, Recall, F1-Score).

> **Important Notice:** This system is an automated text classifier based on stylistic and linguistic patterns. It is **not** an accredited fact-checking service or live truth database.

---

## 2. System Architecture

### Runtime Prediction Architecture
```
User Input (Article / Headline)
             ↓
     Streamlit Interface
             ↓
  Text Preprocessing (Regex, lowercasing, HTML/URL stripping, publisher dateline removal)
             ↓
    TF-IDF Vectorization (50,000 n-gram features)
             ↓
Trained Logistic Regression Pipeline
             ↓
REAL / FAKE Output + Confidence Percentage + Diagnostics
```

### Model Training Lifecycle
```
Kaggle Dataset (Fake.csv + True.csv)
             ↓
Data Cleaning & Deduplication (39,094 clean articles)
             ↓
Stratified Train/Test Split (80% Train, 20% Held-out Test)
             ↓
TF-IDF Vectorization (learned inside Pipeline to prevent data leakage)
             ↓
Model Training (Logistic Regression + Multi-Algorithm Benchmarks)
             ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
             ↓
Joblib Persistence (models/fake_news_pipeline.joblib)
```

---

## 3. Key Features

- **High-Accuracy Classification:** Primary Logistic Regression pipeline achieves **98.62% test accuracy** and **98.73% F1-score** on unseen test data.
- **Robust Preprocessing (FR-02):** Normalizes text by removing HTML markup, URLs, special punctuation, and strips wire-service dateline artifacts (e.g. `(Reuters) -`) to prevent trivial shortcut learning.
- **Probabilistic Confidence Meter (FR-08):** Interprets prediction certainty (`High`, `Moderate`, `Low`) and visualizes class probabilities.
- **Interactive Streamlit Web Dashboard (FR-09):**
  - **Home:** Project purpose, problem statement, architecture workflow.
  - **News Detection:** 1-click test sample buttons, user input text area, instant classification banner, confidence progress bar, and preprocessed text viewer.
  - **Model Performance:** Interactive KPI cards, annotated Confusion Matrix heatmap, detailed Classification Report table, and comparative multi-algorithm benchmark charts.
  - **About:** Comprehensive dataset documentation, technology stack specifications, and ethical caveats.
- **Multi-Algorithm Benchmarks:** Includes side-by-side performance comparison against Multinomial Naive Bayes, Passive Aggressive Classifier, and Linear SVM.
- **Modular & Maintainable Codebase (NFR-04):** Clean separation of concerns across preprocessing, training, evaluation, and prediction.

---

## 4. Project Directory Structure

```
Fake_News_Detection/
├── dataset/
│   ├── Fake.csv                       # Kaggle fake news articles (23,481 records)
│   └── True.csv                       # Kaggle true news articles (21,417 records)
├── models/
│   ├── fake_news_pipeline.joblib      # Serialized scikit-learn TF-IDF + Logistic Regression pipeline
│   ├── evaluation_metrics.json        # Test set evaluation metrics & class reports
│   ├── model_comparisons.json         # Benchmark comparisons across 4 ML algorithms
│   └── confusion_matrix.png           # Annotated confusion matrix plot
├── notebooks/
│   └── exploratory_analysis.ipynb     # Jupyter notebook for exploratory data analysis (EDA)
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── data_preprocessing.py          # Cleaning, labeling, deduplication, dataset preparation
│   ├── train_model.py                 # Pipeline construction, training, benchmark evaluation
│   ├── evaluate_model.py              # Metrics calculation and confusion matrix generation
│   ├── predict.py                     # Inference engine, error validation, confidence scoring
│   └── ui_components.py               # 3D Holographic AI Sphere, dark design system, Plotly charts
├── app.py                             # Streamlit frontend application
├── requirements.txt                   # Project Python dependencies
├── SPEC.md                            # Complete Software Requirements Specification (SRS)
├── README.md                          # Project documentation
└── .gitignore                         # Git exclusion rules
```

---

## 5. Dataset Details

- **Dataset Source:** [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon.
- **Files:**
  - `Fake.csv`: 23,481 articles (Labeled as `0` / `FAKE`)
  - `True.csv`: 21,417 articles (Labeled as `1` / `REAL`)
- **Attributes:**
  | Attribute | Description |
  | :--- | :--- |
  | `title` | News article headline |
  | `text` | Full body text of the article |
  | `subject` | Categorical topic (Politics, News, World News, Government) |
  | `date` | Publication date string |
  | `label` | Assigned binary classification target (`0` = FAKE, `1` = REAL) |

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Virtual environment (recommended)

### Step 1: Clone or Navigate to Directory
```bash
cd /Users/supriyaa/Desktop/ml
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. How to Run the Application

Launch the Streamlit web application with:

```bash
streamlit run app.py
```

The application will start and open automatically in your browser at:
`http://localhost:8501`

---

## 8. Model Training & Benchmarks

To retrain the models from scratch using `dataset/Fake.csv` and `dataset/True.csv`:

```bash
python3 src/train_model.py
```

### Performance on Held-Out Test Set (7,819 Samples)

| Algorithm | Accuracy | Precision (REAL) | Recall (REAL) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **98.62%** | **98.27%** | **99.20%** | **98.73%** |
| **Linear Support Vector Machine** | 99.22% | 99.25% | 99.32% | 99.28% |
| **Passive Aggressive Classifier** | 99.17% | 99.04% | 99.43% | 99.23% |
| **Multinomial Naive Bayes** | 95.81% | 96.82% | 95.40% | 96.10% |

### Primary Model Confusion Matrix (N = 7,819)
- **True Negatives (FAKE identified as FAKE):** 3,506 (44.8%)
- **True Positives (REAL identified as REAL):** 4,205 (53.8%)
- **False Positives (FAKE misclassified as REAL):** 74 (0.9%)
- **False Negatives (REAL misclassified as FAKE):** 34 (0.4%)

---

## 9. Programmatic Usage

You can use the inference engine in your own Python scripts:

```python
from src.predict import predict_news

# Test article
article = """The Senate voted 68 to 32 on Tuesday to pass a bipartisan infrastructure bill
that will fund roads, bridges, public transit, and clean water initiatives across the nation."""

result = predict_news(article)

print(f"Classification : {result['label']}")
print(f"Confidence     : {result['confidence_percent']} ({result['confidence_level']})")
print(f"Probabilities  : {result['probabilities']}")
```

Output:
```text
Classification : REAL
Confidence     : 89.5% (High)
Probabilities  : {'FAKE': 0.1048, 'REAL': 0.8952}
```

---

## 10. Limitations & Ethical Notice

According to Section 12 of the SRS:
1. **Linguistic Correlation vs Ground Truth:** The model learns stylistic, vocabulary, and n-gram distributions from the training corpus. It does not fact-check claims against live reality.
2. **Emerging Misinformation:** Fabricated stories on novel topics or unprecedented events not represented in the historical training data may bypass classification patterns.
3. **Short Headlines:** Very brief headlines (fewer than 5 words) contain limited lexical information.
4. **Advisory Tool:** Model outputs must serve as screening assistance, not as a replacement for human editorial discernment.

---

## 11. Future Enhancements

According to Section 13 of the SRS:
- **Transformer Architectures:** Implement fine-tuned BERT, RoBERTa, or DeBERTa models for contextual representation.
- **Multilingual Support:** Extend tokenization to classify non-English articles.
- **Explainable AI (XAI):** Integrate SHAP and LIME to highlight exact word contributions.
- **URL & Source Credibility:** Incorporate domain age, reputation, and publisher metadata.
- **Fact-Checking APIs:** Connect to Google Fact Check Tools API for cross-verification.
