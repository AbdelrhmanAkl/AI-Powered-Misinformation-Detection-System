```markdown
# AI-Powered Misinformation Detection System

An NLP-based Machine Learning system for binary classification of English news articles as **Real News** or **Fake News**.

The project demonstrates an end-to-end, production-oriented workflow covering TF-IDF feature engineering, calibrated machine learning classification, probability-based decision making, production model packaging, frozen inference configuration, evaluation, and Streamlit deployment.

## Live Demo

[Launch the Streamlit Application](https://ai-powered-misinformation-detection-system-kpwevip8yqegvhzvnrk.streamlit.app/)

## Repository

[GitHub Repository](https://github.com/AbdelrhmanAkl/AI-Powered-Misinformation-Detection-System)

---

## Overview

The **AI-Powered Misinformation Detection System** is an English-language NLP classification project designed to identify patterns associated with real and fake news labels in a news dataset.

The system accepts two inputs:

- News title
- News article body

It then applies the saved production inference pipeline to generate:

- Predicted label
- Fake News probability
- Production decision threshold

The production application uses a previously trained and packaged model. It does **not** retrain the model during inference.

### Classification Labels

| Label | Meaning |
|---|---|
| `0` | Real News |
| `1` | Fake News |

> **Important:** This project is a classification system, not a factual verification or fact-checking system. It does not browse the internet, independently verify claims, or determine objective truth. Predictions reflect patterns learned from the labels present in the training data.

---

## Project Objectives

The project focuses on demonstrating practical Machine Learning and NLP engineering principles, including:

- NLP feature engineering with TF-IDF
- Binary news classification
- Probability calibration
- Production model packaging
- Frozen inference configuration
- Separation of training and inference
- Model evaluation
- Streamlit application deployment
````markdown
## Key Features

### NLP-Based News Classification

The system classifies English news articles into two categories:

- Real News
- Fake News

The prediction is based on the article's `title` and `text`.

### Calibrated Probability Output

The production classifier is based on `CalibratedClassifierCV`, allowing the system to produce a Fake News probability that can be evaluated against the configured production threshold.

### Frozen Production Inference

The production application uses the saved vectorizer and classifier.

During inference, the existing vectorizer is used through:

```python
vectorizer.transform(...)
````

The application does not call:

```python
fit()
```

or:

```python
fit_transform()
```

The Streamlit application therefore does not retrain the model when users submit articles.

### Production Threshold

The production decision threshold is:

```text
0.36
```

This threshold is stored as part of the production model package and is used consistently during inference.

---

## System Architecture

The overall system can be represented as:

```text
                    User Input
                        |
                        v
              +-------------------+
              |   News Title      |
              |   Article Text    |
              +-------------------+
                        |
                        v
              +-------------------+
              | TF-IDF Vectorizer |
              +-------------------+
                        |
                        v
              +----------------------+
              | CalibratedClassifierCV|
              +----------------------+
                        |
                        v
              +----------------------+
              | Fake News Probability|
              +----------------------+
                        |
                        v
              +----------------------+
              | Threshold = 0.36     |
              +----------------------+
                        |
                        v
              +----------------------+
              | Real News / Fake News|
              +----------------------+
```

---

## Production Inference Pipeline

The production inference pipeline is:

```text
News Title + Article Text
        ↓
TF-IDF Vectorization
        ↓
CalibratedClassifierCV
        ↓
Fake News Probability
        ↓
Frozen Production Threshold = 0.36
        ↓
Real News / Fake News
```

### Production Feature Policy

The production model uses:

| Feature   | Production Usage |
| --------- | ---------------- |
| `title`   | Used             |
| `text`    | Used             |
| `subject` | Excluded         |

The `subject` field is intentionally excluded from the production feature set.

---

## Machine Learning Model

### Vectorizer

```text
TfidfVectorizer
```

TF-IDF converts the text representation into numerical features that can be consumed by the machine learning classifier.

### Classifier

```text
CalibratedClassifierCV
```

The classifier produces a probability output that is used by the production decision rule.

### Production Threshold

```text
0.36
```

The threshold is applied to the Fake News probability to determine the final classification.

The threshold is frozen as part of the production configuration and is not recalculated by the Streamlit application.

---

## Training and Evaluation Workflow

The project maintains a separation between:

* Training data
* Validation data
* Test data

The test set should remain independent from model selection and threshold optimization.

Exact split percentages, training hyperparameters, preprocessing procedures, and random seeds are not specified as part of this project documentation and therefore are intentionally not assumed here.

```
```
````markdown
## Dataset

The project uses two dataset files:

```text
data/True.csv
data/Fake.csv
````

### Dataset Schema

| Column    | Description                 |
| --------- | --------------------------- |
| `title`   | News article title          |
| `text`    | News article body           |
| `subject` | Dataset subject information |
| `date`    | Article date                |

### Dataset Size

| File       |   Articles |
| ---------- | ---------: |
| `True.csv` |     21,417 |
| `Fake.csv` |     23,481 |
| **Total**  | **44,898** |

For production inference, only the following fields are used:

```text
title
text
```

The `subject` field is excluded from the production model features.

---

## Production Model Package

The production model is stored at:

```text
models/misinformation_model_package.joblib
```

The Joblib package contains:

```text
model
vectorizer
threshold
config
seed
model_name
feature_policy
label_mapping
metadata
```

Keeping these components together provides a single production artifact containing the model and the configuration required to reproduce its inference behavior.

This approach helps maintain consistency between the trained model and the inference environment by preserving:

* The trained classifier
* The fitted vectorizer
* The production threshold
* The feature policy
* The label mapping
* Relevant model metadata and configuration

As a result, the application does not need to reconstruct these decisions independently during inference.

---

## Evaluation

A **Production Inference Diagnostic** was performed against 1,000 samples:

* 500 Real News samples
* 500 Fake News samples
* 1,000 samples in total

The samples were drawn from the available:

```text
data/True.csv
data/Fake.csv
```

### Diagnostic Results

| Metric    |  Result |
| --------- | ------: |
| Accuracy  |  99.90% |
| Precision | 100.00% |
| Recall    |  99.80% |
| F1 Score  |  99.90% |

### Confusion Matrix

```text
                 Predicted
                 Real   Fake

Actual Real       500     0
Actual Fake         1   499
```

### Evaluation Context

These results are reported specifically as a **Production Inference Diagnostic**.

They must **not** be interpreted as final held-out test performance because the 1,000 diagnostic samples were sampled from the available `True.csv` and `Fake.csv` datasets.

Therefore, the reported:

```text
99.90% Accuracy
```

is **not presented as final test accuracy**.

A reliable estimate of generalization performance should be based on an independent held-out test set and, ideally, additional external validation data.

---

## Streamlit Application

The interactive application is implemented in:

```text
app.py
```

The application allows users to provide:

* News title
* News article body

The application returns a structured prediction containing:

* Predicted label
* Fake News probability
* Production threshold

The application loads the saved production model package and performs inference using the packaged configuration.

### Application Behavior

The Streamlit application:

* Uses the saved vectorizer
* Uses the saved classifier
* Uses the production threshold of `0.36`
* Does not retrain the model
* Does not fit the vectorizer during inference

### Live Application

[Open the deployed Streamlit application](https://ai-powered-misinformation-detection-system-kpwevip8yqegvhzvnrk.streamlit.app/)

---

## Evaluation Script

The production package can also be evaluated using:

```text
evaluate_model.py
```

Run:

```powershell
.\.venv\Scripts\python.exe evaluate_model.py
```

The evaluation script evaluates the saved production model package without retraining it.

```
```
````markdown
## Installation

### Requirements

The project uses:

```text
Python 3.11
````

A local virtual environment can be created using:

```text
.venv
```

Project dependencies are defined in:

```text
requirements.txt
```

### 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/AI-Powered-Misinformation-Detection-System.git
```

### 2. Enter the Project Directory

```bash
cd AI-Powered-Misinformation-Detection-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Depending on the local PowerShell execution policy, script activation may be restricted.

The environment can still be used directly without activation through:

```powershell
.\.venv\Scripts\python.exe
```

### 5. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 7. Run the Evaluation

```powershell
.\.venv\Scripts\python.exe evaluate_model.py
```

---

## Usage

### Run Locally

Start the application with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application provides an interface for entering:

```text
News Title
Article Body
```

The saved production model package is then used to generate the prediction.

The resulting output includes:

```text
Predicted Label
Fake News Probability
Production Threshold
```

The production threshold used by the application is:

```text
0.36
```

No model retraining takes place when a prediction is requested.

---

## Project Structure

```text
AI-Powered-Misinformation-Detection-System/
│
├── app.py
├── evaluate_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── misinformation_model_package.joblib
│
├── data/
│   ├── True.csv
│   └── Fake.csv
│
└── notebooks/
    └── <TRAINING_NOTEBOOK_FILENAME>
```

---

## Limitations

This system should be interpreted as a machine learning classification model rather than a factual verification engine.

### Dataset-Specific Patterns

The model learns patterns associated with the labels present in its underlying dataset. These patterns may not represent the broader characteristics of real-world news.

### Distribution Shift

News topics, writing styles, sources, and language patterns can change over time. A model trained on one dataset may therefore perform differently on future or unseen data.

### Dataset Artifacts

Machine learning models can potentially learn dataset-specific artifacts or correlations rather than the underlying concept of misinformation itself.

### False Positives and False Negatives

No binary classifier is guaranteed to classify every article correctly. Incorrect predictions can occur in both directions.

### Limited Real-World Generalization

Strong performance on a particular evaluation sample does not guarantee equivalent performance on unseen real-world news articles.

### Classification vs. Factual Verification

The system predicts a classification label based on learned patterns. It does not verify whether the claims in an article are objectively true.

It does not:

* Browse the internet to verify claims
* Fact-check individual statements
* Determine objective truth
* Replace professional fact-checking
* Guarantee that an article is factually true or false

### Independent Validation

Independent external validation is required before making claims about performance on broader real-world news distributions.

---

## Responsible Use / Disclaimer

This project is intended for educational, research, and portfolio demonstration purposes.

The prediction should be treated as a machine learning classification result rather than a definitive statement about the factual accuracy of an article.

The system identifies patterns associated with the labels present in its training data. It does not establish objective truth and should not be used as a standalone source for factual verification or high-stakes decisions.

---

## Future Improvements

The following are potential **future improvements** and are not presented as existing functionality:

* Independent external validation
* Temporal validation
* Source robustness analysis
* Dataset leakage and artifact analysis
* Calibration analysis
* Explainability
* Transformer-based model comparison
* Prediction drift monitoring
* Continuous evaluation
* More robust real-world evaluation

These improvements would help assess model robustness and generalization beyond the current production inference diagnostic.

---

## Technologies

The project uses the following technologies:

| Technology             | Role                                  |
| ---------------------- | ------------------------------------- |
| Python                 | Core programming language             |
| Pandas                 | Data handling                         |
| NumPy                  | Numerical computing                   |
| Scikit-learn           | Machine learning                      |
| TfidfVectorizer        | NLP feature representation            |
| CalibratedClassifierCV | Probability-calibrated classification |
| Joblib                 | Production model packaging            |
| Streamlit              | Interactive application               |
| Jupyter / Google Colab | Development and experimentation       |

---

## Author

**Abdelrahman**

Machine Learning / NLP Portfolio Project

---

## Project Summary

The **AI-Powered Misinformation Detection System** demonstrates a complete NLP classification workflow from dataset-based modeling to packaged production inference.

The core production design combines:

```text
News Text
    ↓
TF-IDF
    ↓
CalibratedClassifierCV
    ↓
Fake News Probability
    ↓
Threshold = 0.36
    ↓
Real News / Fake News
```

The model, vectorizer, threshold, feature policy, label mapping, and metadata are stored together in a Joblib production package, allowing the Streamlit application to perform consistent inference without retraining.

The project is intentionally positioned as a **misinformation classification system**, not a fact-checking or truth-verification system.

```
```

The goal is to provide a reproducible portfolio project that demonstrates how a trained NLP model can be packaged and used consistently in an application environment.
```
