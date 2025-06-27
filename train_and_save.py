"""
Run this ONCE to create vectorizer.pkl and model.pkl
$ python train_model.py
"""

import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# ─── 1. download NLTK resources ──────────────────────────────────────────────
nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text: str) -> str:
    """Lowercase -> tokenize -> remove non-alpha -> stem -> stop-word filter."""
    import re, nltk
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    tokens = nltk.word_tokenize(text)
    return " ".join(
        stemmer.stem(tok) for tok in tokens if tok not in stop_words
    )

# ─── 2. tiny demo data (replace with your real dataset) ───────────────────────
data = {
    "text": [
        "Congratulations! You've won a $500 Amazon gift card. Claim here!",
        "Your invoice is attached, please review.",
        "Limited time offer, click the link to get free bitcoins!",
        "Let's meet for coffee tomorrow morning.",
        "URGENT: Your account will be suspended unless you verify now.",
        "Reminder: project meeting at 2 PM today.",
    ],
    "label": [1, 0, 1, 0, 1, 0],  # 1 = spam, 0 = ham
}
df = pd.DataFrame(data)

# ─── 3. pipeline: TF-IDF → Naive Bayes ────────────────────────────────────────
pipe = Pipeline(
    [
        ("tfidf", TfidfVectorizer(preprocessor=preprocess)),
        ("clf", MultinomialNB()),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)
pipe.fit(X_train, y_train)

print(classification_report(y_test, pipe.predict(X_test)))

# ─── 4. save vectorizer + model ──────────────────────────────────────────────
vectorizer: TfidfVectorizer = pipe.named_steps["tfidf"]
model: MultinomialNB = pipe.named_steps["clf"]

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Saved vectorizer.pkl and model.pkl")
