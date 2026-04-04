"""
📰 News Article Classifier — Main Streamlit App
Multi-model classification: ML · DL · Transformer
"""

import streamlit as st
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from auth.login import login_page, logout_button

# ───────────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────────

st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────
# Custom CSS
# ───────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main title styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }

    /* Card style */
    .metric-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.15);
    }
    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: #6366f1;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────
# Auth Gate
# ───────────────────────────────────────────────────

if not login_page():
    st.stop()

# Logged in — show the app
logout_button()

# ───────────────────────────────────────────────────
# Home Page
# ───────────────────────────────────────────────────

st.markdown('<p class="main-title">📰 News Article Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Classify news articles using Machine Learning, Deep Learning, and Transformers</p>', unsafe_allow_html=True)

st.markdown("---")

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">📊</div>
        <div class="metric-label">EDA Dashboard</div>
        <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
            Explore the AG News dataset with interactive visualizations, word clouds, and statistics.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">🤖</div>
        <div class="metric-label">Predict</div>
        <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
            Classify articles in real-time. Choose between ML, DL, or Transformer models.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">📋</div>
        <div class="metric-label">Activity Log</div>
        <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.5rem;">
            Track all user logins, predictions, and model usage in the admin panel.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# Model overview
st.markdown("### 🏗️ Models")

model_data = {
    "Model Family": ["🔢 Machine Learning", "🧠 Deep Learning", "🤖 Transformer"],
    "Algorithm": ["Logistic Regression + TF-IDF", "Bidirectional LSTM", "DistilBERT (fine-tuned)"],
    "Framework": ["scikit-learn", "PyTorch", "HuggingFace Transformers"],
    "Key Feature": ["Fast, interpretable", "Captures sequential context", "State-of-the-art accuracy"],
}
st.table(model_data)

# Categories
st.markdown("### 🏷️ Categories")
cat_cols = st.columns(4)
categories = [
    ("🌍 World", "#6366f1"),
    ("⚽ Sports", "#f43f5e"),
    ("💼 Business", "#22c55e"),
    ("🔬 Sci/Tech", "#f59e0b"),
]
for col, (cat, color) in zip(cat_cols, categories):
    col.markdown(f"""
    <div style="background: {color}20; border: 1px solid {color}40; border-radius: 10px;
                padding: 1rem; text-align: center;">
        <div style="font-size: 1.3rem; font-weight: 600;">{cat}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "👈 Use the sidebar to navigate to EDA, Prediction, or Activity Log pages.</p>",
    unsafe_allow_html=True,
)
