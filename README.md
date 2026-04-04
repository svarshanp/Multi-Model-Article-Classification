# 📰 Multi-Model News Article Classifier

A complete end-to-end news article classification system that classifies articles into **4 categories** using **3 model families**, wrapped in a **Streamlit web application** with user authentication and activity logging.

## 🏷️ Categories

| ID | Category    | Icon |
|----|-------------|------|
| 1  | World       | 🌍   |
| 2  | Sports      | ⚽   |
| 3  | Business    | 💼   |
| 4  | Sci/Tech    | 🔬   |

## 🏗️ Models

| Model Family       | Algorithm                      | Framework              |
|---------------------|-------------------------------|------------------------|
| Machine Learning    | Logistic Regression + TF-IDF  | scikit-learn           |
| Deep Learning       | Bidirectional LSTM            | PyTorch                |
| Transformer         | DistilBERT (fine-tuned)       | HuggingFace Transformers |

## 📂 Project Structure

```
news_classifier/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── README.md                   # This file
│
├── config/
│   └── settings.py             # Centralized configuration
│
├── src/
│   ├── data_preprocessing.py   # Text cleaning, TF-IDF, tokenization
│   ├── eda.py                  # EDA visualizations
│   ├── evaluate.py             # Metrics + confusion matrix
│   ├── predict.py              # Unified prediction interface
│   └── models/
│       ├── ml_model.py         # Logistic Regression + Naive Bayes
│       ├── dl_model.py         # LSTM classifier
│       └── transformer_model.py # DistilBERT fine-tuning
│
├── auth/
│   ├── database.py             # SQLAlchemy ORM + SQLite
│   └── login.py                # Login/register + session management
│
├── pages/
│   ├── 1_📊_EDA.py             # Exploratory Data Analysis dashboard
│   ├── 2_🤖_Predict.py         # Article classification page
│   └── 3_📋_Activity_Log.py    # User activity log viewer
│
├── scripts/
│   ├── download_data.py        # Download AG News dataset
│   ├── train_ml.py             # Train ML models
│   ├── train_dl.py             # Train LSTM model
│   └── train_transformer.py    # Fine-tune DistilBERT
│
└── tests/
    └── test_predict.py         # Prediction pipeline tests
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

```bash
python scripts/download_data.py
```

This downloads the AG News dataset (~120k training + ~7.6k test articles) from HuggingFace.

### 3. Train Models

**ML models (fastest — ~1 minute):**
```bash
python scripts/train_ml.py
```

**LSTM model (~5-15 minutes on CPU):**
```bash
python scripts/train_dl.py
```

**DistilBERT (30-60+ minutes on CPU, or use a subset):**
```bash
# Full training
python scripts/train_transformer.py

# Quick test with 5000 samples
set MAX_TRAIN_SAMPLES=5000
set MAX_TEST_SAMPLES=1000
python scripts/train_transformer.py
```

### 4. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 5. First Use

1. **Register** a new account on the login page
2. **Log in** with your credentials
3. Navigate to **📊 EDA** to explore the dataset
4. Go to **🤖 Predict** to classify articles
5. Check **📋 Activity Log** to see your history

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 🗄️ Database

- **Local development**: SQLite (automatic, stored at `data/app.db`)
- **Production**: PostgreSQL via AWS RDS (set `DATABASE_URL` in `.env`)

### Tables

| Table          | Purpose                              |
|----------------|--------------------------------------|
| `users`        | Username + hashed password           |
| `activity_log` | Login events, predictions, model use |

## ☁️ AWS Deployment (Optional)

1. Copy `.env.example` to `.env` and fill in your AWS credentials
2. Set `DATABASE_URL` to your RDS PostgreSQL connection string
3. Upload model artifacts to S3
4. Deploy to EC2 and run `streamlit run app.py --server.port 8501`

## 📊 Dataset

**AG News** — A benchmark dataset for text classification:
- 120,000 training samples
- 7,600 test samples
- 4 balanced classes
- Source: [HuggingFace Datasets](https://huggingface.co/datasets/ag_news)

## 📝 License

This project is for educational purposes (Final Year Project).
