# SOFTWARE REQUIREMENTS SPECIFICATION (SRS)
## Fake News Detection Using Machine Learning

**Project Type:** Machine Learning / Natural Language Processing  
**Programming Language:** Python  
**Application Framework:** Streamlit  
**Version:** 1.0  

---

## 1. Introduction

### 1.1 Purpose
The purpose of this project is to develop a Fake News Detection application using Machine Learning and Natural Language Processing.
The application analyzes textual news content and classifies it into two categories: **REAL** and **FAKE**.

### 1.2 Problem Statement
The rapid growth of digital media and online news platforms has increased the circulation of misleading and fabricated information.
Manually verifying every news article is difficult and time-consuming.
This project aims to develop a machine learning-based solution that identifies patterns in labeled news articles and assists users in classifying newly submitted news text.

### 1.3 Project Scope
The application will:
- Accept news article text from users.
- Perform text preprocessing.
- Convert text into numerical features.
- Apply a trained machine learning classifier.
- Display the predicted category.
- Provide model evaluation metrics.
- Offer a simple web interface.

*The application is an automated text classifier, not a fact-checking or source-verification service.*

---

## 2. Objectives
The main objectives are:
- Understand and analyze fake and real news datasets.
- Apply NLP techniques to textual data.
- Extract features using TF-IDF.
- Train machine learning classification algorithms.
- Evaluate model performance.
- Develop a user-friendly prediction interface.
- Demonstrate an end-to-end machine learning workflow.

---

## 3. Technology Requirements

| Technology | Purpose |
| :--- | :--- |
| **Python 3** | Core programming |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **NLTK** | NLP preprocessing |
| **Scikit-learn** | Machine learning |
| **TF-IDF** | Feature extraction |
| **Logistic Regression** | Primary classification algorithm |
| **Matplotlib** | Visualization |
| **Seaborn** | Statistical visualization |
| **Streamlit** | Interactive web frontend |
| **Joblib** | Model persistence |

---

## 4. Dataset Requirements

### 4.1 Dataset Source
Kaggle Fake and Real News Dataset:  
[https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### 4.2 Dataset Files
- `Fake.csv`
- `True.csv`

### 4.3 Dataset Attributes
| Attribute | Description |
| :--- | :--- |
| `title` | News headline |
| `text` | News article content |
| `subject` | News category / topic |
| `date` | Article publication date |
| `label` | Classification label (`0` for FAKE, `1` for REAL) |

### 4.4 Dataset Preparation
The system must:
1. Load the dataset files.
2. Assign FAKE and REAL labels.
3. Combine both datasets.
4. Handle missing values.
5. Remove duplicate articles.
6. Shuffle the combined dataset.
7. Inspect class distribution.
8. Split the dataset into training and testing sets (80% train, 20% test stratified).

---

## 5. Functional Requirements

- **FR-01: Dataset Loading**  
  The system shall load the provided fake and real news CSV files.
- **FR-02: Data Preprocessing**  
  The system shall clean text data by removing unnecessary HTML tags, punctuation where appropriate, and extra whitespace.
- **FR-03: Feature Extraction**  
  The system shall use TF-IDF to transform textual news data into numerical feature vectors.
- **FR-04: Model Training**  
  The system shall train a Logistic Regression classifier using labeled news data.
- **FR-05: Model Evaluation**  
  The system shall calculate Accuracy, Precision, Recall, F1-score, and a Confusion matrix.
- **FR-06: Model Saving**  
  The trained preprocessing and classification pipeline shall be saved using Joblib.
- **FR-07: News Prediction**  
  The application shall accept user-provided news text and generate a REAL or FAKE classification.
- **FR-08: Prediction Display**  
  The application shall display the predicted category and an appropriately interpreted confidence estimate.
- **FR-09: Performance Visualization**  
  The application shall display evaluation metrics and a confusion matrix.

---

## 6. Non-Functional Requirements

- **NFR-01: Usability**  
  The application should provide a simple interface that beginners and general users can understand.
- **NFR-02: Performance**  
  The application should return predictions within a reasonable time for ordinary article-sized inputs.
- **NFR-03: Reliability**  
  The system should handle empty inputs, missing model files, and invalid data gracefully.
- **NFR-04: Maintainability**  
  The application should use modular code with separate preprocessing, training, evaluation, and prediction modules.
- **NFR-05: Reproducibility**  
  The project should document random seeds, dataset versions, dependency versions, and evaluation procedures.
- **NFR-06: Security**  
  The application must not execute user-submitted text as code.

---

## 7. System Architecture

```
User Input
    ↓
Streamlit Interface
    ↓
Text Preprocessing
    ↓
TF-IDF Feature Extraction
    ↓
Trained Logistic Regression Model
    ↓
Prediction
    ↓
REAL / FAKE Result
```

Training architecture:
```
Dataset (Fake.csv + True.csv)
    ↓
Data Cleaning & Deduplication
    ↓
Stratified Train-Test Split (80/20)
    ↓
TF-IDF Vectorization
    ↓
Model Training (Logistic Regression + Comparisons)
    ↓
Model Evaluation
    ↓
Save Model Pipeline (Joblib)
```

---

## 8. Machine Learning Methodology

### 8.1 Text Preprocessing
- Lowercase conversion.
- HTML tag removal.
- URL removal.
- Punctuation and bracket removal.
- Whitespace normalization.
- Handling missing text and duplicates.

### 8.2 Feature Extraction
- TF-IDF (Term Frequency-Inverse Document Frequency) conversion.
- Vocabulary parameters: `ngram_range=(1, 2)`, `max_features=50000`, `sublinear_tf=True`.

### 8.3 Classification Algorithm
- **Primary algorithm:** Logistic Regression (`C=1.0`, `solver='lbfgs'`, `max_iter=1000`).
- **Optional comparison algorithms:**
  - Naive Bayes (MultinomialNB)
  - Passive Aggressive Classifier
  - Linear Support Vector Machine

### 8.4 Dataset Splitting
- Training: 80%
- Testing: 20%
- Stratified splitting (`stratify=y`, `random_state=42`) to preserve class distribution.
- All learned preprocessing and vectorization performed within pipeline to prevent data leakage.

### 8.5 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 9. User Interface Requirements

Streamlit application sections:
1. **Home**: Project title, introduction, project description, architecture overview.
2. **News Detection**: News article input area, prediction button, quick test sample buttons, REAL / FAKE badge result, confidence estimate, preprocessing view.
3. **Model Performance**: Accuracy, precision, recall, F1-score cards, confusion matrix plot, classification report, model comparison chart.
4. **About**: Project description, dataset details, technologies used, model limitations, fact-checking disclaimer.

---

## 10. Project Directory Structure

```
Fake_News_Detection/
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   ├── fake_news_pipeline.joblib
│   ├── evaluation_metrics.json
│   └── confusion_matrix.png
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── app.py
├── requirements.txt
├── SPEC.md
├── README.md
└── .gitignore
```

---

## 11. Limitations & Ethical Considerations
- Predictions depend on the quality and representativeness of the training dataset.
- The model may learn writing-style or publisher-related patterns rather than factual truth.
- New or emerging misinformation may not be recognized.
- Short headlines may provide insufficient lexical signals.
- Automated classification is **not a substitute for professional fact-checking**.
- Confidence scores should not be interpreted as absolute factual probabilities.
