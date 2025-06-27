# app.py ───────────────────────────────────────────────────────────
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1️⃣  Page settings – must be FIRST Streamlit call
st.set_page_config(page_title="Spam Classifier", page_icon="🚫")

# 2️⃣  Load NLTK resources once and cache them
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")
    return set(stopwords.words("english")), PorterStemmer()

STOP_WORDS, STEMMER = load_nltk()

# 3️⃣  Load the trained vectorizer and model once
@st.cache_resource
def load_artifacts():
    with open("vectorizer.pkl", "rb") as vf:
        vectorizer = pickle.load(vf)
    with open("model.pkl", "rb") as mf:
        model = pickle.load(mf)
    return vectorizer, model

VECTORIZER, MODEL = load_artifacts()

# 4️⃣  Text-preprocessing identical to your training pipeline
def transform_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = nltk.word_tokenize(text)
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)

# 5️⃣  Streamlit UI
st.title("📧 Email / SMS Spam Classifier")

user_msg = st.text_area("✍️ Enter a message:")

if st.button("Predict"):
    if not user_msg.strip():
        st.warning("⚠️  Please enter some text.")
    else:
        cleaned = transform_text(user_msg)
        vector = VECTORIZER.transform([cleaned])
        pred = MODEL.predict(vector)[0]

        if pred == 1:
            st.error("🚫 Spam")
        else:
            st.success("✅ Not Spam")
