import os
import string

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="Emotion Analyzer", page_icon="💭", layout="centered")

@st.cache_resource
def load_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))

@st.cache_resource
def setup_tokenizer():
    try:
        word_tokenize("test")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

stop_words = load_stop_words()
setup_tokenizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = "".join(char for char in text if not char.isdigit())
    text = "".join(char for char in text if char.isascii())
    words = word_tokenize(text)
    return " ".join(word for word in words if word not in stop_words)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "SVM": "svm_model.pkl",
}
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

# Emotion mapping from the notebook's label encoding.
EMOTION_LABELS = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy",
}

@st.cache_resource
def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)

st.title("💭 Emotion Analyzer")
st.write("Enter a sentence and the trained NLP model will predict the emotion associated with it.")
st.caption("Current pipeline: text preprocessing → TF-IDF → classifier")
st.divider()

with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Choose model", list(MODEL_FILES.keys()))
    st.info("The notebook currently contains Logistic Regression and SVM models trained using TF-IDF.")
    st.divider()
    st.caption("The notebook currently uses numeric emotion classes (0–5).")

model = load_model(MODEL_FILES[selected_model])
vectorizer = load_model(VECTORIZER_FILE)

text = st.text_area("Enter your text", placeholder="Example: I am feeling really happy today!", height=160)
analyze = st.button("Analyze Emotion", type="primary", use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Please enter some text first.")
        st.stop()

    if model is None:
        st.error(f"{selected_model} model file was not found: {MODEL_FILES[selected_model]}")
        st.info("Save the trained model in the same directory as app.py.")
        st.stop()

    if vectorizer is None:
        st.error("TF-IDF vectorizer file was not found: tfidf_vectorizer.pkl")
        st.info("Save the fitted TfidfVectorizer in the same directory as app.py.")
        st.stop()

    cleaned_text = preprocess_text(text)

    if not cleaned_text.strip():
        st.warning("No usable words remained after preprocessing. Please enter more text.")
        st.stop()

    try:
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        st.success("Prediction completed!")
        emotion_name = EMOTION_LABELS.get(
            int(prediction),
            f"Unknown (Class {prediction})",
        )

        st.subheader("Predicted Emotion")
        st.metric("Emotion", emotion_name)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vectorized_text)[0]
            probability = probabilities[list(model.classes_).index(prediction)] * 100
            st.subheader("Prediction Confidence")
            st.progress(min(max(probability / 100, 0.0), 1.0))
            st.write(f"**{probability:.2f}%**")

            with st.expander("Show all class probabilities"):
                for class_label, class_probability in zip(model.classes_, probabilities):
                    class_name = EMOTION_LABELS.get(
                        int(class_label),
                        f"Unknown (Class {class_label})",
                    )
                    st.write(
                        f"{class_name}: {class_probability * 100:.2f}%"
                    )

        with st.expander("Show preprocessed text"):
            st.write(cleaned_text)

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)

st.divider()
st.caption("This is an NLP machine-learning project. Predictions should not be treated as a psychological or medical diagnosis.")
