# Deployment Document — Multi-Model News Article Classifier

> **Project**: News Article Classification System  
> **Version**: 1.0  
> **Date**: April 2026  
> **Author**: Final Year Project  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements](#2-system-requirements)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Local Development Setup](#4-local-development-setup)
5. [Model Training Guide](#5-model-training-guide)
6. [AWS Cloud Deployment](#6-aws-cloud-deployment)
7. [Database Configuration](#7-database-configuration)
8. [Environment Variables](#8-environment-variables)
9. [Security Considerations](#9-security-considerations)
10. [Monitoring & Maintenance](#10-monitoring--maintenance)
11. [Troubleshooting](#11-troubleshooting)
12. [Rollback Procedure](#12-rollback-procedure)

---

## 1. Project Overview

### 1.1 Purpose

This application classifies news articles into four categories — **World**, **Sports**, **Business**, and **Sci/Tech** — using three distinct model families:

| Model Family       | Algorithm                     | Framework              | Typical Accuracy |
|--------------------|-------------------------------|------------------------|------------------|
| Machine Learning   | Logistic Regression + TF-IDF  | scikit-learn           | ~92%             |
| Deep Learning      | Bidirectional LSTM            | PyTorch                | ~90-92%          |
| Transformer        | DistilBERT (fine-tuned)       | HuggingFace Transformers | ~94%           |

### 1.2 Key Features

- Multi-model prediction with real-time confidence scores
- User authentication (registration + login with bcrypt hashing)
- Interactive EDA dashboard with word clouds, charts, and statistics
- User activity logging (logins, predictions, model usage)
- Streamlit-based web interface

### 1.3 Technology Stack

| Layer          | Technology                          |
|----------------|-------------------------------------|
| Frontend       | Streamlit 1.45+                     |
| Backend Logic  | Python 3.11+                        |
| ML Framework   | scikit-learn, PyTorch, HuggingFace  |
| Database       | SQLite (local) / PostgreSQL (prod)  |
| Cloud Platform | AWS (EC2 + S3 + RDS)                |
| OS Target      | Ubuntu 22.04 LTS (EC2)              |

---

## 2. System Requirements

### 2.1 Minimum Hardware (Local Development)

| Resource | Minimum          | Recommended         |
|----------|------------------|---------------------|
| CPU      | 4 cores          | 8+ cores            |
| RAM      | 8 GB             | 16 GB               |
| Storage  | 5 GB free        | 20 GB free          |
| GPU      | Not required     | CUDA-enabled (for Transformer training) |

### 2.2 Minimum Hardware (Production — AWS EC2)

| Resource   | Instance Type    | Specs                              |
|------------|------------------|------------------------------------|
| ML only    | `t3.medium`      | 2 vCPU, 4 GB RAM                   |
| ML + LSTM  | `t3.large`       | 2 vCPU, 8 GB RAM                   |
| All models | `t3.xlarge`      | 4 vCPU, 16 GB RAM                  |
| Training   | `g4dn.xlarge`    | 4 vCPU, 16 GB RAM, NVIDIA T4 GPU   |

### 2.3 Software Prerequisites

| Software      | Version   | Purpose                    |
|---------------|-----------|----------------------------|
| Python        | 3.11+     | Runtime                    |
| pip           | 23.0+     | Package management         |
| Git           | 2.30+     | Version control            |
| AWS CLI       | 2.x       | Cloud deployment           |
| PostgreSQL    | 15+       | Production database        |

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT BROWSER                        │
│                     (http://your-ip:8501)                    │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTPS (via reverse proxy)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     AWS EC2 INSTANCE                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              NGINX (Reverse Proxy)                     │  │
│  │              Port 80/443 → localhost:8501              │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              ▼                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            STREAMLIT APPLICATION (app.py)              │  │
│  │                   Port 8501                            │  │
│  │  ┌─────────┐  ┌──────────┐  ┌───────────────┐        │  │
│  │  │  Login   │  │   EDA    │  │   Prediction  │        │  │
│  │  │  Module  │  │ Dashboard│  │    Engine      │        │  │
│  │  └────┬────┘  └──────────┘  └───────┬───────┘        │  │
│  │       │                             │                  │  │
│  │       ▼                             ▼                  │  │
│  │  ┌─────────┐               ┌────────────────┐         │  │
│  │  │ Auth DB │               │  Model Loader   │         │  │
│  │  │(SQLAlch)│               │  ML / DL / TF   │         │  │
│  │  └────┬────┘               └───────┬────────┘         │  │
│  └───────│────────────────────────────│──────────────────┘  │
│          │                            │                      │
│          ▼                            ▼                      │
│  ┌──────────────┐           ┌──────────────────┐            │
│  │  PostgreSQL   │           │  Model Artifacts  │            │
│  │  (AWS RDS)    │           │  (AWS S3 Bucket)  │            │
│  └──────────────┘           └──────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Local Development Setup

### 4.1 Clone and Install

```bash
# Clone the repository
git clone <your-repo-url> news_classifier
cd news_classifier

# Create a virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4.2 Download Dataset

```bash
python scripts/download_data.py
```

This downloads the AG News dataset (120,000 train + 7,600 test samples) from HuggingFace.

### 4.3 Train Models

```bash
# ML models (fastest — ~1 minute)
python scripts/train_ml.py

# LSTM model (~5-15 minutes on CPU)
python scripts/train_dl.py

# DistilBERT (30-60+ minutes on CPU)
python scripts/train_transformer.py

# Quick DistilBERT test with subset
set MAX_TRAIN_SAMPLES=5000
set MAX_TEST_SAMPLES=1000
python scripts/train_transformer.py
```

### 4.4 Run Locally

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`.

### 4.5 Run Tests

```bash
python -m pytest tests/ -v
```

---

## 5. Model Training Guide

### 5.1 Training Pipeline

```
Raw CSV Data
    │
    ▼
┌──────────────────────┐
│  data_preprocessing   │
│  • clean_text()       │
│  • TF-IDF / Tokenize  │
│  • HF Tokenizer       │
└──────────┬───────────┘
           │
    ┌──────┼──────────┐
    ▼      ▼          ▼
┌──────┐ ┌──────┐ ┌──────────┐
│  ML  │ │ LSTM │ │DistilBERT│
│ .pkl │ │ .pt  │ │  model/  │
└──────┘ └──────┘ └──────────┘
    │      │          │
    ▼      ▼          ▼
  models/  models/   models/
    ml/      dl/    transformer/
```

### 5.2 Model Artifact Locations

| Model             | Artifacts                                           | Size (approx.) |
|--------------------|-----------------------------------------------------|----------------|
| Logistic Regression| `models/ml/logistic_regression.pkl` + `tfidf_vectorizer.pkl` | ~150 MB |
| Naive Bayes        | `models/ml/naive_bayes.pkl`                        | ~50 MB         |
| LSTM               | `models/dl/lstm_model.pt` + `vocab.pkl`             | ~25 MB         |
| DistilBERT         | `models/transformer/distilbert_final/` (directory)  | ~250 MB        |

### 5.3 Retraining

To retrain a model with new data:
1. Replace `data/raw/train.csv` and `data/raw/test.csv` with your updated data
2. Ensure the CSV format is: `label, title, description` (labels 1-4)
3. Run the corresponding training script
4. Restart the Streamlit app to load the new model

---

## 6. AWS Cloud Deployment

### 6.1 Prerequisites

- AWS account with IAM permissions for EC2, S3, and RDS
- AWS CLI configured (`aws configure`)
- A key pair for SSH access to EC2
- A domain name (optional, for HTTPS)

### 6.2 Step 1 — Launch EC2 Instance

```bash
# Launch Ubuntu 22.04 instance (t3.large recommended)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type t3.large \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=news-classifier}]'
```

**Security Group Rules:**

| Type  | Protocol | Port   | Source      | Purpose             |
|-------|----------|--------|-------------|---------------------|
| SSH   | TCP      | 22     | Your IP     | Admin access        |
| HTTP  | TCP      | 80     | 0.0.0.0/0   | Web traffic         |
| HTTPS | TCP      | 443    | 0.0.0.0/0   | Secure web traffic  |
| Custom| TCP      | 8501   | 0.0.0.0/0   | Streamlit (direct)  |

### 6.3 Step 2 — Configure EC2 Instance

SSH into the instance and run:

```bash
#!/bin/bash
# ========================================
# EC2 Setup Script for News Classifier
# ========================================

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11+ and essentials
sudo apt install -y python3 python3-pip python3-venv git nginx

# Clone your project
cd /home/ubuntu
git clone <your-repo-url> news_classifier
cd news_classifier

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your production values:
# nano .env
```

### 6.4 Step 3 — Set Up PostgreSQL (AWS RDS)

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier news-classifier-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 15 \
  --master-username admin \
  --master-user-password YOUR_STRONG_PASSWORD \
  --allocated-storage 20 \
  --publicly-accessible \
  --db-name news_classifier

# Wait for it to become available
aws rds wait db-instance-available \
  --db-instance-identifier news-classifier-db

# Get the endpoint
aws rds describe-db-instances \
  --db-instance-identifier news-classifier-db \
  --query 'DBInstances[0].Endpoint.Address' \
  --output text
```

Update `.env` on EC2:
```bash
DATABASE_URL=postgresql://admin:YOUR_STRONG_PASSWORD@your-rds-endpoint:5432/news_classifier
```

Install the PostgreSQL driver:
```bash
pip install psycopg2-binary
```

### 6.5 Step 4 — Upload Models to S3

```bash
# Create S3 bucket
aws s3 mb s3://news-classifier-models --region us-east-1

# Upload model artifacts
aws s3 sync models/ s3://news-classifier-models/models/ --exclude "*.pyc"

# Download on EC2
aws s3 sync s3://news-classifier-models/models/ models/
```

### 6.6 Step 5 — Configure Streamlit as a Systemd Service

Create `/etc/systemd/system/streamlit.service`:

```ini
[Unit]
Description=News Classifier Streamlit App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/news_classifier
Environment="PATH=/home/ubuntu/news_classifier/venv/bin"
ExecStart=/home/ubuntu/news_classifier/venv/bin/python -m streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.maxUploadSize 50 \
  --browser.gatherUsageStats false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl start streamlit
sudo systemctl status streamlit
```

### 6.7 Step 6 — Configure NGINX Reverse Proxy

Create `/etc/nginx/sites-available/news-classifier`:

```nginx
server {
    listen 80;
    server_name your-domain.com;   # or use the EC2 public IP

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    # WebSocket support (required for Streamlit)
    location /_stcore/stream {
        proxy_pass http://localhost:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/news-classifier /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### 6.8 Step 7 — Add HTTPS with Let's Encrypt (Optional)

```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
sudo systemctl restart nginx
```

---

## 7. Database Configuration

### 7.1 Schema

```sql
-- Users Table
CREATE TABLE users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    VARCHAR(80) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Activity Log Table
CREATE TABLE activity_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    VARCHAR(80) NOT NULL,
    action      VARCHAR(50) NOT NULL,    -- 'login', 'predict', 'logout'
    model_used  VARCHAR(50),             -- 'ml', 'dl', 'transformer'
    input_text  TEXT,
    prediction  VARCHAR(50),
    confidence  VARCHAR(20),
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_activity_username ON activity_log(username);
CREATE INDEX idx_users_username ON users(username);
```

### 7.2 Migration from SQLite to PostgreSQL

Tables are auto-created by SQLAlchemy on startup. To migrate existing data:

```bash
# Export from SQLite
sqlite3 data/app.db ".dump users" > users_dump.sql
sqlite3 data/app.db ".dump activity_log" > activity_dump.sql

# Import to PostgreSQL (adjust SQL syntax if needed)
psql -h your-rds-endpoint -U admin -d news_classifier < users_dump.sql
psql -h your-rds-endpoint -U admin -d news_classifier < activity_dump.sql
```

---

## 8. Environment Variables

| Variable               | Required | Default                    | Description                     |
|------------------------|----------|----------------------------|---------------------------------|
| `DATABASE_URL`         | No       | `sqlite:///./data/app.db`  | Database connection string      |
| `SECRET_KEY`           | Yes (prod) | `dev-secret-change-me`   | App secret for session security |
| `AWS_ACCESS_KEY_ID`    | No       | —                          | AWS credentials for S3          |
| `AWS_SECRET_ACCESS_KEY`| No       | —                          | AWS credentials for S3          |
| `AWS_REGION`           | No       | `us-east-1`               | AWS region                      |
| `S3_BUCKET_NAME`       | No       | `news-classifier-models`  | S3 bucket for model artifacts   |
| `DEBUG`                | No       | `True`                     | Debug mode flag                 |
| `MAX_TRAIN_SAMPLES`    | No       | `0` (all)                  | Limit training samples          |
| `MAX_TEST_SAMPLES`     | No       | `0` (all)                  | Limit test samples              |

---

## 9. Security Considerations

### 9.1 Authentication

- Passwords are hashed using **bcrypt** with automatic salt generation
- Session management via Streamlit's `st.session_state` (server-side)
- No plaintext passwords are stored anywhere

### 9.2 Production Hardening Checklist

| Item                                | Status  | Action Required                           |
|-------------------------------------|---------|-------------------------------------------|
| Change `SECRET_KEY`                 | ⬜      | Set a strong random string in `.env`      |
| Use HTTPS                           | ⬜      | Configure Let's Encrypt (Section 6.8)    |
| Restrict Security Group             | ⬜      | Limit SSH to your IP only                |
| Disable debug mode                  | ⬜      | Set `DEBUG=False` in `.env`              |
| Set strong DB password              | ⬜      | Use 20+ character password for RDS       |
| Enable RDS encryption               | ⬜      | Enable at-rest encryption in RDS config  |
| Regular backups                     | ⬜      | Enable automated RDS snapshots           |
| Rotate AWS keys                     | ⬜      | Use IAM roles on EC2 instead of keys     |
| Input sanitization                  | ✅      | `clean_text()` strips HTML/scripts       |

### 9.3 Network Security

```
Internet → NGINX (80/443) → Streamlit (8501, localhost only)
                                  │
                                  ├── RDS (5432, VPC internal)
                                  └── S3 (HTTPS, IAM auth)
```

- Streamlit should **only listen on localhost** in production
- NGINX handles all external traffic and provides HTTPS termination
- RDS should be in the same VPC as EC2, not publicly accessible

---

## 10. Monitoring & Maintenance

### 10.1 Health Checks

```bash
# Check if Streamlit is running
sudo systemctl status streamlit

# Check logs
sudo journalctl -u streamlit -f --no-pager -n 50

# Check NGINX status
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log

# Check disk usage (model files can be large)
df -h
du -sh models/*
```

### 10.2 Useful Commands

```bash
# Restart the app after code changes
sudo systemctl restart streamlit

# Reload NGINX after config changes
sudo nginx -t && sudo systemctl reload nginx

# View active database connections (PostgreSQL)
psql -h your-rds-endpoint -U admin -d news_classifier \
  -c "SELECT * FROM pg_stat_activity WHERE datname='news_classifier';"

# View recent activity logs
psql -h your-rds-endpoint -U admin -d news_classifier \
  -c "SELECT * FROM activity_log ORDER BY timestamp DESC LIMIT 20;"
```

### 10.3 Log Rotation

Add to `/etc/logrotate.d/streamlit`:

```
/var/log/streamlit/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 ubuntu ubuntu
}
```

### 10.4 Updating the Application

```bash
cd /home/ubuntu/news_classifier

# Pull latest code
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Update models from S3 (if changed)
aws s3 sync s3://news-classifier-models/models/ models/

# Restart
sudo systemctl restart streamlit
```

---

## 11. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | venv not activated | `source venv/bin/activate` |
| Port 8501 already in use | Previous instance running | `sudo lsof -i :8501` then `kill <PID>` |
| Models not loading | Artifacts missing | Re-run training scripts or sync from S3 |
| Database connection refused | Wrong `DATABASE_URL` | Verify `.env` and RDS security group |
| NGINX 502 Bad Gateway | Streamlit not running | `sudo systemctl start streamlit` |
| Slow predictions | Model loading on each request | Models are cached with `@st.cache_resource` |
| Unicode errors on Windows | cp1252 encoding | Scripts include `sys.stdout.reconfigure(encoding='utf-8')` |
| Out of memory during training | Insufficient RAM | Use `MAX_TRAIN_SAMPLES` env var for smaller batches |

### Debug Mode

```bash
# Run Streamlit with debug output
streamlit run app.py --logger.level debug 2>&1 | tee debug.log
```

---

## 12. Rollback Procedure

### 12.1 Code Rollback

```bash
cd /home/ubuntu/news_classifier

# View recent commits
git log --oneline -10

# Rollback to a specific commit
git checkout <commit-hash>

# Restart the app
sudo systemctl restart streamlit
```

### 12.2 Model Rollback

```bash
# S3 versioning should be enabled for model rollbacks
aws s3api list-object-versions \
  --bucket news-classifier-models \
  --prefix models/ml/logistic_regression.pkl

# Restore a specific version
aws s3api get-object \
  --bucket news-classifier-models \
  --key models/ml/logistic_regression.pkl \
  --version-id <version-id> \
  models/ml/logistic_regression.pkl

sudo systemctl restart streamlit
```

### 12.3 Database Rollback

```bash
# Restore from RDS automated snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier news-classifier-db-restored \
  --db-snapshot-identifier <snapshot-id>
```

---

## Deployment Checklist

Use this checklist before going live:

- [ ] All models trained and tested locally
- [ ] `.env` configured with production values
- [ ] `SECRET_KEY` set to a strong random value
- [ ] `DEBUG=False` in production
- [ ] EC2 instance launched and configured
- [ ] Security groups properly restricted
- [ ] RDS database created and accessible from EC2
- [ ] `DATABASE_URL` points to RDS
- [ ] Model artifacts uploaded to S3 and synced to EC2
- [ ] Streamlit service running (`systemctl status streamlit`)
- [ ] NGINX configured and serving traffic
- [ ] HTTPS enabled (Let's Encrypt)
- [ ] Test login, EDA, prediction, and activity log pages
- [ ] Automated backups enabled (RDS snapshots)
- [ ] Monitoring set up (CloudWatch / logs)

---

*Document generated for the Multi-Model News Article Classifier — Final Year Project.*
