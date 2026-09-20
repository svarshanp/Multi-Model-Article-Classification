<img width="1519" height="816" alt="{A7DB85B3-4B10-4741-9553-F5C2BA57C276}" src="https://github.com/user-attachments/assets/704c0c9d-b127-46bd-bc39-b01b5c31a1a6" />
<img width="1461" height="816" alt="{0BBDC169-F6F3-439B-9D0C-6C2483BAF6F4}" src="https://github.com/user-attachments/assets/60746efb-cbb7-4dcc-a59b-61df89757c81" />
<img width="1480" height="827" alt="{A961F895-007C-4FDF-BE17-B28A1C5BD02B}" src="https://github.com/user-attachments/assets/2fd425b2-ffc9-466f-bf7c-c84002e8f846" />
<img width="1502" height="819" alt="{5CA754C1-55A6-45D5-871D-5F23FC40C2DA}" src="https://github.com/user-attachments/assets/b21a74b1-72c9-49df-a400-ecdd087f7b05" />
<img width="1448" height="782" alt="{35E9DD3B-2EA8-4002-9F3E-04C36BF4E859}" src="https://github.com/user-attachments/assets/cd8961e4-3b3e-4f46-8edf-f00357c98e52" />
# Multi-Model Article Classification — Implementation Plan

## Goal

Build a complete end-to-end news article classification system that:
1. Classifies articles into 4 categories (World, Sports, Business, Sci/Tech)
2. Uses 3 model families: ML (Logistic Regression), DL (LSTM), Transformer (DistilBERT)
3. Exposes a Streamlit web app with secure login and model selection
4. Deploys on AWS (EC2 + S3 + RDS) with user activity logging

---

## User Review Required

> [!IMPORTANT]
> **AWS Services**: The project requires AWS EC2, S3, and RDS. Do you already have an AWS account set up? If not, we can build everything locally first and add AWS integration later.

> [!IMPORTANT]
> **Deep Learning Framework**: Your system has neither PyTorch nor TensorFlow installed. I recommend **PyTorch** (better HuggingFace integration). Shall I install it?

> [!IMPORTANT]
> **Dataset**: The AG News dataset needs to be downloaded from Kaggle. Do you have a Kaggle account / API key, or should I download it manually?

> [!WARNING]
> **Database for local dev**: For local development I'll use **SQLite** (zero setup), with a clear migration path to AWS RDS (PostgreSQL) for production. Is that acceptable?

---

## Proposed Directory Structure

```
shree/
├── app.py                      # Main Streamlit application
├── requirements.txt            # All Python dependencies
├── setup.sh                    # AWS deployment script
├── README.md                   # Full documentation
├── .env.example                # Template for environment variables
│
├── data/
│   ├── raw/                    # Original AG News CSV files
│   └── processed/              # Cleaned/preprocessed data
│
├── notebooks/
│   └── eda.ipynb               # Exploratory Data Analysis (optional)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Text cleaning, tokenization, vectorization
│   ├── eda.py                  # EDA plots & visualizations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ml_model.py         # Logistic Regression / Naive Bayes
│   │   ├── dl_model.py         # LSTM / GRU
│   │   └── transformer_model.py # DistilBERT fine-tuning
│   ├── evaluate.py             # Metrics: accuracy, F1, confusion matrix
│   └── predict.py              # Unified prediction interface
│
├── models/                     # Saved model artifacts
│   ├── ml/                     # .pkl files (TF-IDF vectorizer + classifier)
│   ├── dl/                     # .pt files (LSTM/GRU weights)
│   └── transformer/            # .bin / model directory (DistilBERT)
│
├── pages/
│   ├── 1_📊_EDA.py             # Streamlit EDA dashboard page
│   ├── 2_🤖_Predict.py         # Prediction page (model selector + input)
│   └── 3_📋_Activity_Log.py    # User activity log viewer (admin)
│
├── auth/
│   ├── __init__.py
│   ├── login.py                # Login form & session management
│   └── database.py             # SQLAlchemy models + DB connection
│
├── config/
│   └── settings.py             # App settings, AWS config, DB URLs
│
├── scripts/
│   ├── train_ml.py             # Script to train ML models
│   ├── train_dl.py             # Script to train DL models
│   ├── train_transformer.py    # Script to train/fine-tune DistilBERT
│   └── download_data.py        # Download AG News from Kaggle
│
└── tests/
    └── test_predict.py         # Basic prediction tests
```

---

## Proposed Changes

### Phase 1 — Environment & Data Setup

#### [NEW] [requirements.txt](file:///Users/dineshmurugaiyan/Desktop/shree/requirements.txt)
All project dependencies:
- `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`
- `scikit-learn` (ML models + TF-IDF)
- `torch`, `torchtext` (DL models)
- `transformers`, `datasets`, `accelerate` (HuggingFace)
- `sqlalchemy`, `psycopg2-binary` (Database)
- `boto3` (AWS S3 integration)
- `python-dotenv` (environment variables)

#### [NEW] [scripts/download_data.py](file:///Users/dineshmurugaiyan/Desktop/shree/scripts/download_data.py)
- Downloads AG News dataset from Kaggle (or uses a direct URL fallback)
- Saves to `data/raw/train.csv` and `data/raw/test.csv`

---

### Phase 2 — Data Preprocessing & EDA

#### [NEW] [src/data_preprocessing.py](file:///Users/dineshmurugaiyan/Desktop/shree/src/data_preprocessing.py)
- `clean_text()`: Remove HTML, URLs, emojis, extra whitespace
- `preprocess_for_ml()`: Lowercase → tokenize → remove stopwords → TF-IDF
- `preprocess_for_dl()`: Tokenize → sequence padding (fixed length)
- `preprocess_for_transformer()`: HuggingFace AutoTokenizer encoding

#### [NEW] [src/eda.py](file:///Users/dineshmurugaiyan/Desktop/shree/src/eda.py)
- Class distribution bar chart
- Word clouds per category
- Average word count per article
- Title/description length distribution
- Heatmap of frequent words per class

---

### Phase 3 — Model Building

#### [NEW] [src/models/ml_model.py](file:///Users/dineshmurugaiyan/Desktop/shree/src/models/ml_model.py)
- Train Logistic Regression + Naive Bayes on TF-IDF features
- Save: TF-IDF vectorizer (`.pkl`) + trained model (`.pkl`)
- Functions: `train()`, `predict()`, `evaluate()`

#### [NEW] [src/models/dl_model.py](file:///Users/dineshmurugaiyan/Desktop/shree/src/models/dl_model.py)
- PyTorch LSTM model with:
  - Embedding layer → LSTM → Fully connected → Softmax
  - Dropout for regularization
- Save: vocabulary mapping + model weights (`.pt`)
- Functions: `LSTMClassifier` class, `train()`, `predict()`

#### [NEW] [src/models/transformer_model.py](file:///Users/dineshmurugaiyan/Desktop/shree/src/models/transformer_model.py)
- Fine-tune `distilbert-base-uncased` using HuggingFace Trainer API
- 4-class classification head on top of DistilBERT
- Save: full model directory (tokenizer + model weights)
- Functions: `train()`, `predict_with_confidence()`

#### [NEW] [src/evaluate.py](file:///Users/dineshmurugaiyan/Desktop/shree/src/evaluate.py)
- Compute Accuracy, Precision, Recall, F1-score per model
- Generate confusion matrix plots
- Side-by-side comparison table

---

### Phase 4 — Streamlit Application

#### [NEW] [app.py](file:///Users/dineshmurugaiyan/Desktop/shree/app.py)
- Main entry point: login gate → redirect to prediction page
- Streamlit page config, custom CSS theming
- Session state management

#### [NEW] [auth/login.py](file:///Users/dineshmurugaiyan/Desktop/shree/auth/login.py)
- Login form (username + password)
- Session-based authentication using `st.session_state`
- Log login events to database

#### [NEW] [auth/database.py](file:///Users/dineshmurugaiyan/Desktop/shree/auth/database.py)
- SQLAlchemy ORM models:
  - `User` table: id, username, password_hash
  - `LoginActivity` table: id, username, timestamp, selected_model
- SQLite for local dev → PostgreSQL (RDS) for production
- Connection factory with env-var-based URL

#### [NEW] [pages/1_📊_EDA.py](file:///Users/dineshmurugaiyan/Desktop/shree/pages/1_📊_EDA.py)
- Interactive EDA dashboard showing visualizations from `src/eda.py`
- Requires login

#### [NEW] [pages/2_🤖_Predict.py](file:///Users/dineshmurugaiyan/Desktop/shree/pages/2_🤖_Predict.py)
- Model selector: ML / DL / Transformer (dropdown)
- Text area for article input
- "Predict" button → shows predicted category + confidence score
- Logs activity to database

#### [NEW] [pages/3_📋_Activity_Log.py](file:///Users/dineshmurugaiyan/Desktop/shree/pages/3_📋_Activity_Log.py)
- Displays user activity from database in a table
- Filterable by user and date

#### [NEW] [src/predict.py](file:///Users/dineshmurugaiyan/Desktop/shree/src/predict.py)
- Unified prediction interface that loads any of the 3 model types
- Returns: predicted class label + confidence score
- Handles model caching with `@st.cache_resource`

---

### Phase 5 — AWS Deployment

#### [NEW] [config/settings.py](file:///Users/dineshmurugaiyan/Desktop/shree/config/settings.py)
- Centralized configuration: DB URL, S3 bucket, model paths
- Reads from environment variables (`.env`) with sensible defaults

#### [NEW] [setup.sh](file:///Users/dineshmurugaiyan/Desktop/shree/setup.sh)
- EC2 deployment script: install dependencies, configure Streamlit, start app
- Systemd service file for auto-restart

#### [NEW] [.env.example](file:///Users/dineshmurugaiyan/Desktop/shree/.env.example)
- Template for: `DATABASE_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`

#### [NEW] [README.md](file:///Users/dineshmurugaiyan/Desktop/shree/README.md)
- Project overview, setup instructions, architecture diagram
- How to train models, run locally, deploy to AWS
- Database schema documentation

---

## Open Questions

> [!IMPORTANT]
> 1. **AWS Account**: Do you have an active AWS account? If not, I'll build everything to run locally first (SQLite + local file storage) and add AWS integration hooks so you can switch later.

> [!IMPORTANT]
> 2. **Kaggle Dataset**: Do you have a Kaggle API key configured, or should I provide a direct download alternative?

> [!IMPORTANT]
> 3. **Model Training**: Training DistilBERT can take **30–60+ minutes on CPU**. Are you okay with CPU-based training, or do you have access to a GPU (CUDA)?

> [!NOTE]
> 4. **Scope Priority**: Given the 14-day timeline, should I focus on getting a **fully working local version first** and then layer AWS deployment on top? This is the approach I recommend.

---

## Verification Plan

### Automated Tests
1. Run `python -m pytest tests/` to verify prediction pipeline
2. Run `streamlit run app.py` and manually test login → predict → log flow
3. Verify all 3 models produce predictions with confidence scores

### Manual Verification
1. **Login flow**: Create account → login → verify session persists across pages
2. **Prediction accuracy**: Test with sample articles from each category
3. **Model comparison**: Verify all 3 model types are selectable and produce results
4. **Activity logging**: Check database tables contain login + prediction records
5. **Browser testing**: Walk through the full user flow in browser and record it

### Deployment Verification (when AWS is configured)
1. Upload model artifacts to S3 → verify app loads them
2. Verify RDS connection and data persistence
3. Access app via EC2 public IP
