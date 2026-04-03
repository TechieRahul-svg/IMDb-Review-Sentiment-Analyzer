import streamlit as st
import pickle
import re
import random

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load Model + Vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_artifacts()

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(review):
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[1] if prediction == 1 else probability[0]

    return cleaned, sentiment, probability, confidence

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(255, 221, 120, 0.35) 0%, rgba(255, 221, 120, 0) 35%),
            radial-gradient(circle at 90% 0%, rgba(120, 212, 255, 0.30) 0%, rgba(120, 212, 255, 0) 40%),
            linear-gradient(180deg, #fffdf5 0%, #f3faff 100%);
        color: #172033;
    }

    .hero-card {
        background:
            linear-gradient(100deg, rgba(245, 197, 24, 0.18) 0%, rgba(245, 197, 24, 0.00) 28%),
            linear-gradient(160deg, #ffffff 0%, #f6fbff 100%);
        padding: 30px;
        border-radius: 30px;
        border: 1px solid #dbeafe;
        box-shadow: 0 16px 38px rgba(16, 24, 40, 0.10);
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }

    .hero-card::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.28) 0%, rgba(56, 189, 248, 0) 70%);
        right: -55px;
        top: -55px;
        pointer-events: none;
    }

    .hero-topline {
        display: inline-block;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7c2d12;
        background: #ffedd5;
        border: 1px solid #fdba74;
        border-radius: 999px;
        padding: 0.3rem 0.75rem;
        margin-bottom: 0.75rem;
    }

    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.4rem;
    }

    .hero-clap {
        width: 56px;
        height: 56px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #fcd34d;
        font-size: 1.8rem;
        box-shadow: 0 6px 16px rgba(245, 158, 11, 0.22);
        flex-shrink: 0;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #f5c518;
        line-height: 1.1;
    }

    .hero-sub {
        font-size: 1.05rem;
        color: #334155;
        line-height: 1.7;
        max-width: 920px;
        margin-bottom: 1rem;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .hero-pill {
        font-size: 0.82rem;
        font-weight: 700;
        color: #0f172a;
        background: #eef6ff;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: 0.3rem 0.7rem;
    }

    .metric-card {
        padding: 18px;
        border-radius: 20px;
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 10px 26px rgba(30, 41, 59, 0.08);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 4px;
        background: #cbd5e1;
    }

    .metric-gold {
        background: linear-gradient(180deg, #ffffff 0%, #fffaf0 100%);
    }

    .metric-gold::before {
        background: linear-gradient(90deg, #f59e0b 0%, #facc15 100%);
    }

    .metric-sky {
        background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%);
    }

    .metric-sky::before {
        background: linear-gradient(90deg, #06b6d4 0%, #38bdf8 100%);
    }

    .metric-coral {
        background: linear-gradient(180deg, #ffffff 0%, #fff1f2 100%);
    }

    .metric-coral::before {
        background: linear-gradient(90deg, #fb7185 0%, #f97316 100%);
    }

    .metric-icon {
        font-size: 1.3rem;
        margin-bottom: 2px;
    }

    .metric-title {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 5px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.35rem;
        font-weight: 700;
    }

    .result-positive {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border: 2px solid #10b981;
        padding: 20px;
        border-radius: 18px;
        color: #064e3b;
        font-size: 1.25rem;
        font-weight: 800;
        box-shadow: 0 8px 24px rgba(16,185,129,0.20);
    }

    .result-negative {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        padding: 20px;
        border-radius: 18px;
        color: #7f1d1d;
        font-size: 1.25rem;
        font-weight: 800;
        box-shadow: 0 8px 24px rgba(239,68,68,0.20);
    }

    .section-head {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 10px;
    }

    .small-note {
        color: #475569;
        font-size: 0.92rem;
        text-align: center;
        margin-top: 30px;
    }

    .sample-btns button {
        width: 100%;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1rem !important;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
    }

    .stButton > button:hover {
        border-color: #93c5fd !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.16);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        border-right: 1px solid #dbeafe;
    }

    section[data-testid="stSidebar"] * {
        color: #1e293b;
    }

    .stTextArea textarea {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
    }

    .stTextArea textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 1px #60a5fa;
    }

    .stAlert {
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🎞 IMDb Sentiment Analyzer")
    st.write(
        "This app predicts whether an **IMDb-style movie review** is **Positive** or **Negative** "
        "using your trained **TF-IDF + Logistic Regression** model."
    )
    st.markdown("---")
    st.markdown("### ⚙ Model Info")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features:** TF-IDF")
    st.write("**Task:** Binary Sentiment Classification")
    st.markdown("---")
    st.markdown("### 🧠 Tips")
    st.write("- Use **movie review style** text")
    st.write("- Longer reviews usually give better signal")
    st.write("- Mixed reviews may produce lower confidence")

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero-card">
    <div class="hero-topline">Now Showing</div>
    <div class="hero-title-row">
        <div class="hero-clap">🎬</div>
        <div class="hero-title">IMDb Sentiment Analyzer</div>
    </div>
    <div class="hero-sub">
        Paste a movie review and instantly classify it as <b>Positive</b> or <b>Negative</b>.
        Built for IMDb-style sentiment analysis using a classical ML baseline.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">Instant Prediction</span>
        <span class="hero-pill">TF-IDF + Logistic Regression</span>
        <span class="hero-pill">IMDb Review Tone</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Top Metrics Row
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="metric-card metric-gold">
        <div class="metric-icon">🎞</div>
        <div class="metric-title">Model Type</div>
        <div class="metric-value">ML Baseline</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card metric-sky">
        <div class="metric-icon">🧮</div>
        <div class="metric-title">Approach</div>
        <div class="metric-value">TF-IDF + LR</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card metric-coral">
        <div class="metric-icon">🍿</div>
        <div class="metric-title">Use Case</div>
        <div class="metric-value">IMDb Reviews</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# -----------------------------
# Sample Reviews
# -----------------------------
positive_samples = [
    "Absolutely brilliant movie! The acting, screenplay, and emotional depth were outstanding. I loved every moment of it.",
    "A masterpiece of storytelling with powerful performances and a beautiful soundtrack.",
    "This film was amazing from start to finish. The plot was engaging and the acting was top-notch.",
    "One of the best movies I have watched in years. Emotional, entertaining, and beautifully directed.",
    "An excellent movie with a gripping storyline and unforgettable characters."
]

negative_samples = [
    "One of the worst films I have seen. The plot was boring, the acting felt forced, and it was a complete waste of time.",
    "Terrible movie. Poor screenplay, weak acting, and an ending that made no sense.",
    "I expected much more, but this film was dull, slow, and painfully disappointing.",
    "A boring and predictable movie with no emotional impact at all.",
    "The acting was awful and the story dragged endlessly. I regret watching it."
]

mixed_samples = [
    "The cinematography was beautiful and the music was great, but the story felt too slow and predictable.",
    "Some performances were excellent, but the screenplay was weak and the pacing was uneven.",
    "The movie had a strong first half, but the second half lost momentum badly.",
    "Visually stunning film, but the emotional connection with the characters was missing.",
    "There were a few great scenes, but overall the movie felt average and forgettable."
]

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.4, 1])

with left:
    st.markdown('<div class="section-head">✍ Enter Review</div>', unsafe_allow_html=True)
    review = st.text_area(
        "Write or paste a movie review below:",
        height=240,
        placeholder="Example: The film starts slowly, but the performances and emotional ending make it absolutely worth watching..."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        predict_btn = st.button("🎯 Predict Sentiment", use_container_width=True)
    with c2:
        clear_btn = st.button("🧹 Clear / Reset", use_container_width=True)

    if clear_btn:
        st.session_state.pop("sample_text", None)
        st.rerun()

with right:
    st.markdown('<div class="section-head">🎥 Try Sample Reviews</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-btns">', unsafe_allow_html=True)

    if st.button("🌟 Sample Positive Review", use_container_width=True):
        st.session_state["sample_text"] = random.choice(positive_samples)

    if st.button("💥 Sample Negative Review", use_container_width=True):
        st.session_state["sample_text"] = random.choice(negative_samples)

    if st.button("🎭 Sample Mixed Review", use_container_width=True):
        st.session_state["sample_text"] = random.choice(mixed_samples)

    st.markdown("</div>", unsafe_allow_html=True)

    if "sample_text" in st.session_state:
        st.markdown('<div class="section-head">📌 Selected Sample Review</div>', unsafe_allow_html=True)
        st.info(st.session_state["sample_text"])

# If sample selected, overwrite review only when input is empty
if "sample_text" in st.session_state and not review:
    review = st.session_state["sample_text"]

# -----------------------------
# Prediction Block
# -----------------------------
if predict_btn:
    if review.strip() == "":
        st.warning("Please enter a movie review before predicting.")
    else:
        cleaned, sentiment, probability, confidence = predict_sentiment(review)

        st.write("")
        st.markdown('<div class="section-head">📊 Prediction Result</div>', unsafe_allow_html=True)

        if sentiment == "Positive":
            st.markdown(
                f'<div class="result-positive">✅ Predicted Sentiment: {sentiment} | Confidence: {confidence*100:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-negative">❌ Predicted Sentiment: {sentiment} | Confidence: {confidence*100:.2f}%</div>',
                unsafe_allow_html=True
            )

        st.write("")
        a, b = st.columns(2)

        with a:
            st.markdown('<div class="section-head">👍 Positive Score</div>', unsafe_allow_html=True)
            st.progress(float(probability[1]))
            st.write(f"**{probability[1]*100:.2f}%**")

        with b:
            st.markdown('<div class="section-head">👎 Negative Score</div>', unsafe_allow_html=True)
            st.progress(float(probability[0]))
            st.write(f"**{probability[0]*100:.2f}%**")

        st.write("")

        with st.expander("🧼 Cleaned Review Text"):
            st.write(cleaned)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    '<div class="small-note">Built for IMDb-style sentiment analysis using your trained TF-IDF + Logistic Regression model.</div>',
    unsafe_allow_html=True
)