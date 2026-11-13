# 🛡️ PhishGuard: Multi-Source Email Threat Classification

**An End-to-End Automated Data Alignment, NLP Feature Engineering, and Real-Time Phishing Detection Pipeline**

*Developed for **CT-264: Programming for Artificial Intelligence** (Complex Computing Activity)*

**Department of Computer Science & Information Technology — NED University of Engineering & Technology**

[📖 Read Technical Report](docs/Technical_Report.md) • [📊 View EDA Notebook](notebooks/01_EDA_and_Cleaning.ipynb) • [🚀 Launch Web App](#-quick-start--installation)

## 📌 Executive Overview

Phishing remains the foremost initial access vector in modern cybersecurity breaches. However, machine learning classifiers trained on isolated, single-source datasets (such as Enron or Nazario alone) routinely fail in production due to **domain overfitting**, **class imbalance**, and **schema incompatibility**.

**PhishGuard** solves this by engineering an automated multi-corpus ingestion and classification pipeline. It dynamically resolves schema conflicts across **seven major benchmark datasets**, deduplicates and sanitizes **163,868 verified emails**, and pairs **sublinear Unigram+Bigram TF-IDF vectorization** with an $L_2$**-regularized Logistic Regression (SAGA)** classifier—achieving **98.60% accuracy** and **98.97% threat recall** in under **5 seconds** of training time.

## ✨ Key Engineering Highlights

* 🔄 **Dynamic Schema Standardization (`src/data_loader.py`):** Automatically inspects heterogeneous CSV headers (`body`, `text_combined`, `message`, `spam`, `label`), applies context-aware substring heuristics with negative guards (blocking false matches on `sender` or `urls`), and handles corrupted SMTP byte streams via tiered encoding fallbacks (`utf-8` $\rightarrow$ `latin1`).
* 🧹 **Large-Scale Data Hygiene:** Consolidates **164,972 raw records** down to **163,868 pristine emails** by purging null/whitespace payloads and eliminating **1,098 cross-corpus duplicate templates**, resulting in a naturally balanced **51.9% Threat / 48.1% Benign** distribution without synthetic SMOTE distortion.
* 🧠 **High-Speed Linear NLP Architecture (`src/preprocess.py` & `src/train.py`):** Extracts $10,000$ sublinear TF-IDF unigram and bigram features with strict train/test split isolation ($80/20$ stratified) to prevent data leakage, converging in **4.9 seconds** with $<5\text{ ms}$ single-email inference latency.
* 🖥️ **Interactive Threat Triage Dashboard (`src/app.py`):** A production-ready **Streamlit** web interface featuring `@st.cache_resource` memory optimization, real-time confidence progress bars, and a stateful rotating benchmark suite of 10+ adversarial phishing and corporate legitimate scenarios.

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------+
|                   RAW MULTI-SOURCE CORPORA (7 CSVs)                   |
|   CEAS_08 | Enron | Ling | Nazario | Nigerian_Fraud | Kaggle | SpamA  |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|             DYNAMIC SCHEMA ALIGNMENT & HYGIENE PIPELINE               |
|  • Priority Column Heuristic Mapping  • Tiered Encoding Auto-Recovery |
|  • Null Purging & Deduplication       • Output: 163,868 Verified Rows |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                STRATIFIED DATA PARTITIONING (80 / 20)                 |
|        Train Split: 131,094 Emails   |   Test Split: 32,774 Emails    |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                  SUBLINEAR TF-IDF FEATURE EXTRACTION                  |
|   10,000 Max Features | Unigrams + Bigrams (1,2) | Stopwords Removed  |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|             LOGISTIC REGRESSION CLASSIFIER (SAGA, L2, C=1.0)          |
|         Training Time: 4.9s  |  Artifact Persistence (.joblib)        |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|               STREAMLIT REAL-TIME INFERENCE DASHBOARD                 |
|      Live Text Triage | Rotating Sample Bank | Confidence Scoring     |
+-----------------------------------------------------------------------+
```

## 📊 Dataset Aggregation & Exploratory Data Analysis

PhishGuard unifies seven distinct public corpora into a single standardized schema (`text`, `label`):

| Dataset Source                   | Raw Rows          | Original Schema Columns                                | Primary Domain Focus                            |
| -------------------------------- | ----------------- | ------------------------------------------------------ | ----------------------------------------------- |
| **`phishing_email.csv`** | 82,486            | `text_combined, label`                               | Aggregated Kaggle benchmark collection          |
| **`CEAS_08.csv`**        | 39,154            | `sender, receiver, date, subject, body, label, urls` | CEAS 2008 Live Spam & Phishing Challenge        |
| **`Enron.csv`**          | 29,767            | `subject, body, label`                               | Enterprise corporate communications baseline    |
| **`SpamAssasin.csv`**    | 5,809             | `sender, receiver, date, subject, body, label, urls` | Apache open-source spam/ham filtering corpus    |
| **`Nigerian_Fraud.csv`** | 3,332             | `sender, receiver, date, subject, body, urls, label` | Advance-fee fraud (419 social engineering)      |
| **`Ling.csv`**           | 2,859             | `subject, body, label`                               | Academic linguistic mailing list baseline       |
| **`Nazario.csv`**        | 1,565             | `sender, receiver, date, subject, body, urls, label` | Targeted credential harvesting & spear-phishing |
| **Total Consolidated**     | **163,868** | **`text, label` (post-deduplication)**         | **Balanced Multi-Vector Threat Corpus**   |

### Visual Insights from `notebooks/01_EDA_and_Cleaning.ipynb`

* **Natural Class Equilibrium:** The merged corpus achieves a **51.9% Phishing (85,053)** to **48.1% Legitimate (78,815)** ratio, eliminating the need for synthetic oversampling (SMOTE).
* **Length & Urgency Profiling:** Phishing emails are markedly shorter on average (**median 428 characters / 66 words**) compared to legitimate emails (**median 908 characters / 143 words**), reflecting concise psychological urgency prompts.

## 🛡️ Model Performance & Evaluation

Evaluated on **32,774 unseen test emails**, the classifier demonstrates exceptional threat interception with minimal false alarms:

| Evaluation Metric                          | Score            | Operational Interpretation                                                |
| ------------------------------------------ | ---------------- | ------------------------------------------------------------------------- |
| **Test Accuracy**                    | **98.60%** | Overall correct classifications across 32,774 test instances              |
| **Recall (Phishing Detection Rate)** | **98.97%** | Intercepts nearly **99 out of every 100** malicious phishing emails |
| **Precision (Phishing)**             | **98.34%** | High reliability when flagging an email as a threat                       |
| **F1-Score (Phishing)**              | **0.9865** | Harmonic balance between precision and recall                             |
| **False Positive Rate (FPR)**        | **1.81%**  | Only 285 benign emails misclassified out of 15,763                        |
| **False Negative Rate (FNR)**        | **1.03%**  | Only 175 phishing emails missed out of 17,011                             |

## 📂 Repository Structure

```
PhishGuard/
├── data/
│   ├── processed/
│   │   └── .gitkeep                        # Target directory for cleaned_phishing_data.csv
│   └── raw/
│       └── .gitkeep                        # Drop raw Kaggle CSV files here before running loader
├── docs/
│   ├── assets/
│   │   ├── class_distribution.png          # Class balance bar chart
│   │   ├── confusion_matrix.png            # Test set evaluation heatmap
│   │   ├── email_length_distribution.png   # Log-scale KDE length distribution
│   │   └── top_words_comparison.png        # Top 20 discriminative unigrams per class
│   ├── PhishGuard_Report.docx              # Formatted Word Technical Report
│   ├── PhishGuard_Report.pdf               # Final 13-Page Printable PDF Report
│   └── Technical_Report.md                 # Markdown source of the Technical Report
├── models/
│   ├── logistic_regression_model.joblib    # Persisted SAGA Logistic Regression classifier
│   └── tfidf_vectorizer.joblib             # Persisted 10,000-feature TF-IDF vectorizer
├── notebooks/
│   └── 01_EDA_and_Cleaning.ipynb           # Interactive statistical analysis & visualization
├── src/
│   ├── __init__.py
│   ├── app.py                              # Streamlit real-time web interface
│   ├── data_loader.py                      # Multi-CSV schema alignment & deduplication
│   ├── preprocess.py                       # Stratified splitting & TF-IDF extraction
│   └── train.py                            # Model training, evaluation & artifact persistence
├── .gitattributes
├── .gitignore
├── requirements.txt                        # Project dependencies
└── README.md
```

## 🚀 Quick Start & Installation

### 1. Clone the Repository & Initialize Environment

```
git clone https://github.com/RazaJavaid2004/PhishGuard.git
cd PhishGuard

# Create and activate an isolated local virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Web App Immediately

Pre-trained model artifacts (`*.joblib`) are included in the `models/` directory, so you can launch the **Streamlit** dashboard right away:

```
.\.venv\Scripts\streamlit run src/app.py
```

> 🌐 The application will automatically open in your browser at **`http://localhost:8501`**.

### 3. Reproduce the Full Data & Training Pipeline *(Optional)*

If you wish to re-run the entire pipeline from scratch:

1. Download the [Phishing Email Detection Dataset](https://www.kaggle.com/datasets/manjeetkumar27/phishing-email-detection-dataset) from Kaggle and extract the 7 CSV files into `data/raw/`.
2. Execute the alignment and training scripts sequentially:

```
# Step 1: Align schemas, clean records, and merge into data/processed/cleaned_phishing_data.csv
python src/data_loader.py

# Step 2: Extract TF-IDF features, train Logistic Regression, and export metrics + artifacts
python src/train.py
```
