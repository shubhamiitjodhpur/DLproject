import streamlit as st
import numpy as np
import requests
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download helper
def download_file_from_huggingface(url, filename):
    if not os.path.exists(filename):
        st.write(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

# Hugging Face URLs (replace with your actual URLs)
BASE_URL = "https://huggingface.co/shubhamprabhukhanolkar/mental-health-sentiment-models/resolve/main/"
model_urls = {
    'LSTM': BASE_URL + 'lstm_sentiment_model.keras',
    'BiLSTM': BASE_URL + 'bilstm_sentiment_model.keras',
    'CNN-BiLSTM': BASE_URL + 'cnn_bilstm_mental_health_sentiment_model.keras',
    'Multi-View': BASE_URL + 'multi_view_hierarchical_sentiment_model.keras'
}

tokenizer_url = BASE_URL + 'tokenizer.pickle'

classes = ['Normal', 'Depression', 'Suicidal', 'Anxiety','Bipolar','Stress','Personality disorder']
    
MAX_SEQUENCE_LENGTH = 150


def download_file_from_huggingface(url, filename):
    """Download a file from Hugging Face if it doesn't exist."""
    if not os.path.exists(filename):
        st.write(f"📥 Downloading {filename}...")
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

def preprocess_text(text):
    """Minimal preprocessing. Replace with your full preprocessing if needed."""
    text = text.lower()
    return text

def remove_stopwords_simple(text):
    """Placeholder for stopword removal. Customize if needed."""
    return text

def predict_sentiment(text, model):
    """Predict sentiment from text using the selected model."""
    processed_text = preprocess_text(text)
    processed_text = remove_stopwords_simple(processed_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction = model.predict(padded_sequence)[0]
    top_class_index = np.argmax(prediction)
    top_class = classes[top_class_index]
    confidence = prediction[top_class_index]
    return top_class, confidence, dict(zip(classes, prediction.tolist()))

# --------------- APP SETUP ---------------

st.title("🧠 Mental Health Sentiment Prediction App")

# Download tokenizer
with st.spinner('🔄 Loading tokenizer...'):
    download_file_from_huggingface(tokenizer_url, 'tokenizer.pickle')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# Download and load models
models = {}
with st.spinner('🔄 Loading models...'):
    for name, url in model_urls.items():
        filename = url.split("/")[-1]
        download_file_from_huggingface(url, filename)
        models[name] = load_model(filename, compile=False)  # 👈 IMPORTANT: compile=False

st.success("✅ Models and Tokenizer Loaded!")

# --------------- APP INTERFACE ---------------

user_input = st.text_area("📝 Enter your sentiment text here:", height=150)
model_choice = st.selectbox("🤖 Select a Model:", list(models.keys()))

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to predict!")
    else:
        with st.spinner('🔮 Predicting sentiment...'):
            model = models[model_choice]
            predicted_class, confidence, all_probs = predict_sentiment(user_input, model)
        
        st.success(f"🎯 Predicted Sentiment: **{predicted_class}** (Confidence: {confidence:.2f})")
        
        st.subheader("🔍 All Class Probabilities:")
        st.bar_chart(all_probs)
