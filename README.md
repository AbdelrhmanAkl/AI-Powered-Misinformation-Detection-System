# AI-Powered Misinformation Detection System

An end-to-end **NLP-based Machine Learning system** for binary classification of English news articles as **Real News** or **Fake News**.

The project demonstrates a production-oriented workflow covering **TF-IDF feature engineering, calibrated classification, probability-based decision making, model packaging, frozen inference configuration, evaluation, and Streamlit deployment**.

> **Important:** This system is a machine learning classification model, not a fact-checking engine. It does not independently verify claims or determine objective truth.

---

## 🚀 Live Demo

**Try the deployed application:**

https://ai-powered-misinformation-detection-system-kpwevip8yqegvhzvnrk.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/AbdelrhmanAkl/AI-Powered-Misinformation-Detection-System

---

## 📌 Project Overview

The **AI-Powered Misinformation Detection System** is an English-language NLP classification application designed to identify patterns associated with real and fake news articles based on a labeled news dataset.

The application accepts:

* **News Title**
* **News Article Body**

The saved production inference pipeline then performs:

```text
User Input
    ↓
Text Combination
    ↓
TF-IDF Vectorization
    ↓
Calibrated Classifier
    ↓
Fake News Probability
    ↓
Production Threshold
    ↓
Real News / Fake News
```

The production application uses a **pre-trained and packaged model**. No model retraining takes place during inference.

---

## 🎯 Project Objectives

The project was developed to demonstrate practical Machine Learning and NLP engineering principles, including:

* Natural Language Processing with TF-IDF
* Binary text classification
* Probability calibration
* Production model packaging
* Frozen inference configuration
* Separation of training and inference
* Probability-based decision making
* Model evaluation
* Streamlit application development
* Local and cloud deployment

---

## ✨ Key Features

### 📰 NLP-Based News Classification

Classifies English news articles into two categories:

| Label | Classification |
| ----: | -------------- |
|   `0` | Real News      |
|   `1` | Fake News      |

The prediction is based on the article:

* `title`
* `text`

---

### 📊 Calibrated Probability Output

The production classifier uses:

```text
CalibratedClassifierCV
```

This enables the application to produce a probability estimate for the **Fake News** class.

The probability is then evaluated against the frozen production threshold.

---

### 🔒 Frozen Production Inference

The Streamlit application loads the saved production model package containing the trained classifier and fitted vectorizer.

During inference, the application uses:

```python
vectorizer.transform(...)
```

It does **not** call:

```python
vectorizer.fit(...)
```

or:

```python
vectorizer.fit_transform(...)
```

Therefore, submitting new articles through the application does not retrain or modify the vectorizer.

---

### 🎚️ Production Decision Threshold

The production decision threshold is:

```text
0.36
```

The threshold is stored inside the production model package and is reused consistently during inference.

The application does not recalculate or optimize the threshold when making predictions.

---

## 🏗️ System Architecture

```text
                         User
                          │
                          ▼
                ┌───────────────────┐
                │   News Title      │
                │   Article Body    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Text Processing │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ TfidfVectorizer   │
                └─────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │ CalibratedClassifierCV  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Fake News Probability   │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Production Threshold    │
              │        0.36             │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Real News / Fake News   │
              └─────────────────────────┘
```

---

## 🔄 Production Inference Pipeline

The complete production inference flow is:

```text
News Title + Article Text
            ↓
      Text Processing
            ↓
      TF-IDF Vectorization
            ↓
    CalibratedClassifierCV
            ↓
     Fake News Probability
            ↓
   Frozen Threshold = 0.36
            ↓
      Final Prediction
            ↓
   Real News / Fake News
```

---

## 🧩 Production Feature Policy

The production model intentionally uses only the following features:

| Feature   | Production Usage |
| --------- | ---------------- |
| `title`   | ✅ Used           |
| `text`    | ✅ Used           |
| `subject` | ❌ Excluded       |
| `date`    | ❌ Excluded       |

The `subject` field is excluded from the production feature set to maintain consistency with the packaged inference configuration.

---

## 🤖 Machine Learning Model

### TF-IDF Vectorization

The system uses:

```text
TfidfVectorizer
```

TF-IDF transforms textual information into numerical feature representations that can be processed by the machine learning classifier.

The fitted vectorizer is stored inside the production package and reused during inference.

---

### Probability-Calibrated Classification

The production classifier is:

```text
CalibratedClassifierCV
```

The classifier provides probability estimates that are used by the production decision rule.

The final classification is determined using the configured Fake News probability threshold:

```text
Threshold = 0.36
```

---

## 📚 Dataset

The project uses two labeled news datasets:

```text
data/True.csv
data/Fake.csv
```

### Dataset Schema

| Column    | Description                 |
| --------- | --------------------------- |
| `title`   | News article title          |
| `text`    | News article body           |
| `subject` | Dataset subject information |
| `date`    | Article date                |

### Dataset Size

| Dataset    |   Articles |
| ---------- | ---------: |
| `True.csv` |     21,417 |
| `Fake.csv` |     23,481 |
| **Total**  | **44,898** |

For production inference, only:

```text
title
text
```

are used as model features.

---

## 📦 Production Model Package

The production model is stored as:

```text
models/misinformation_model_package.joblib
```

The Joblib artifact packages the components required for consistent inference:

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

Keeping these components together creates a single production artifact containing both the trained model and the configuration required to reproduce its inference behavior.

This helps preserve consistency across:

* Model inference
* Feature transformation
* Decision threshold
* Feature selection
* Label mapping
* Model metadata
* Application deployment

The Streamlit application therefore does not need to reconstruct these decisions independently.

---

## 📈 Evaluation

A **Production Inference Diagnostic** was performed using:

* 500 Real News samples
* 500 Fake News samples
* 1,000 samples in total

The samples were drawn from:

```text
data/True.csv
data/Fake.csv
```

### Diagnostic Results

| Metric    |      Result |
| --------- | ----------: |
| Accuracy  |  **99.90%** |
| Precision | **100.00%** |
| Recall    |  **99.80%** |
| F1 Score  |  **99.90%** |

### Confusion Matrix

```text
                    Predicted
                  Real      Fake

Actual Real        500        0
Actual Fake          1      499
```

---

### ⚠️ Evaluation Interpretation

The results above are explicitly reported as a:

> **Production Inference Diagnostic**

They should **not** be interpreted as final held-out test performance.

The 1,000 samples were sampled from the available `True.csv` and `Fake.csv` datasets. Therefore, the reported:

```text
99.90% Accuracy
```

is not presented as an estimate of independent real-world generalization.

A reliable assessment of generalization should use:

* An independent held-out test set
* External validation data
* Ideally, temporal or source-based validation

---

## 🖥️ Streamlit Application

The interactive application is implemented in:

```text
app.py
```

The application allows users to enter:

```text
News Title
Article Body
```

The saved production model package is then loaded to generate the prediction.

### Application Output

The application returns:

* Predicted classification
* Fake News probability
* Production decision threshold

### Application Behavior

The Streamlit application:

* Loads the saved production model
* Loads the fitted TF-IDF vectorizer
* Uses the frozen production threshold
* Performs inference only
* Does not retrain the classifier
* Does not fit the vectorizer
* Does not modify the production configuration

---

## 🌐 Live Application

The deployed application is available here:

https://ai-powered-misinformation-detection-system-kpwevip8yqegvhzvnrk.streamlit.app/

---

## 🧪 Evaluation Script

The saved production package can also be evaluated using:

```text
evaluate_model.py
```

Run:

```powershell
.\.venv\Scripts\python.exe evaluate_model.py
```

The evaluation script evaluates the existing production model package without retraining it.

---

## ⚙️ Installation

### Requirements

The project uses:

```text
Python 3.11
```

Dependencies are defined in:

```text
requirements.txt
```

---

### 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/AI-Powered-Misinformation-Detection-System.git
```

---

### 2. Enter the Project Directory

```bash
cd AI-Powered-Misinformation-Detection-System
```

---

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

---

### 4. Activate the Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, the virtual environment can still be used directly:

```powershell
.\.venv\Scripts\python.exe
```

---

### 5. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 6. Run the Streamlit Application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

### 7. Run the Evaluation

```powershell
.\.venv\Scripts\python.exe evaluate_model.py
```

---

## 🧑‍💻 Usage

### Local Application

Start Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then provide:

```text
News Title
Article Body
```

The application loads the production model package and performs inference.

The result includes:

```text
Predicted Label
Fake News Probability
Production Threshold
```

The production threshold is:

```text
0.36
```

No model retraining occurs during prediction.

---

## 📁 Project Structure

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

## ⚠️ Limitations

### Dataset-Specific Patterns

The model learns statistical patterns associated with the labels in its underlying dataset. These patterns may not fully represent the broader characteristics of real-world news.

### Distribution Shift

News topics, writing styles, publishers, and linguistic patterns can change over time.

Consequently, performance on future or unseen distributions may differ from the diagnostic results reported in this repository.

### Dataset Artifacts

The classifier may learn dataset-specific artifacts, stylistic patterns, or correlations that are associated with the labels rather than the underlying concept of misinformation.

### False Positives and False Negatives

No binary classification model is guaranteed to classify every article correctly.

Incorrect predictions can occur for both Real News and Fake News.

### Limited Real-World Generalization

Strong performance on the current diagnostic sample does not guarantee equivalent performance on:

* Unseen news sources
* Future articles
* Different publishing environments
* Different writing styles
* Real-world misinformation campaigns

### Classification vs. Fact Verification

This system is a **classification model**, not a factual verification engine.

It does not:

* Browse the internet to verify claims
* Fact-check individual statements
* Determine objective truth
* Validate sources independently
* Replace professional fact-checking
* Guarantee that an article is factually true or false

---

## 🛡️ Responsible Use & Disclaimer

This project is intended primarily for:

* Educational purposes
* Machine Learning research
* NLP experimentation
* Portfolio demonstration

Predictions should be treated as **model outputs**, not definitive statements about the factual accuracy of an article.

The system identifies patterns associated with labels present in its training data. It does not establish objective truth.

The application should therefore **not be used as a standalone source for factual verification or high-stakes decisions**.

---

## 🔮 Future Improvements

The following are potential future improvements and are **not currently presented as implemented functionality**:

* Independent external validation
* Temporal validation
* Cross-source robustness analysis
* Dataset leakage analysis
* Dataset artifact analysis
* Probability calibration analysis
* Model explainability
* Transformer-based model comparison
* Prediction drift monitoring
* Continuous evaluation
* Real-world robustness testing
* Additional model architectures
* Production monitoring

---

## 🛠️ Technologies

| Technology                 | Purpose                               |
| -------------------------- | ------------------------------------- |
| **Python**                 | Core programming language             |
| **Pandas**                 | Data manipulation and processing      |
| **NumPy**                  | Numerical computing                   |
| **Scikit-learn**           | Machine Learning                      |
| **TfidfVectorizer**        | Text feature engineering              |
| **CalibratedClassifierCV** | Probability-calibrated classification |
| **Joblib**                 | Model serialization and packaging     |
| **Streamlit**              | Interactive web application           |
| **Jupyter / Google Colab** | Development and experimentation       |

---

## 👨‍💻 Author

**Abdelrahman Akl**

Machine Learning / NLP Portfolio Project

---

## 📌 Project Summary

The **AI-Powered Misinformation Detection System** demonstrates an end-to-end NLP Machine Learning workflow, from dataset-based text classification to packaged production inference and Streamlit deployment.

The core production pipeline is:

```text
News Title + Article Text
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

The production model, fitted vectorizer, threshold, feature policy, label mapping, and metadata are stored together inside a Joblib production package.

This design allows the Streamlit application to perform **consistent, reproducible inference without retraining the model**.

The project is intentionally positioned as a **misinformation classification system rather than a fact-checking or truth-verification system**.

---

## ⭐ Final Note

This project demonstrates how a traditional NLP pipeline can be transformed into a reusable production-oriented Machine Learning application by combining:

```text
Data
  ↓
NLP Feature Engineering
  ↓
Machine Learning
  ↓
Probability Calibration
  ↓
Production Packaging
  ↓
Frozen Inference
  ↓
Streamlit Deployment
```

The focus is not only on achieving a strong diagnostic score, but also on building a clear separation between **model development, production packaging, inference, evaluation, and deployment**.
