import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load models
models = {
    'LSTM': load_model('mental_health_sentiment_model.h5'),
    'BiLSTM': load_model('best_sentiment_model.h5'),
    'CNN-BiLSTM': load_model('cnn_bilstm_mental_health_sentiment_model.h5'),
    'Multi-View': load_model('multi_view_hierarchical_sentiment_model.h5')
}

MAX_SEQUENCE_LENGTH = 150
custom_stopwords = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "im", "ive", "t", "m", "s", "ve", "u"
]

# Preprocessing functions
def preprocess_text(text):
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    text = text.lower()  # Convert to lowercase

    # Remove Markdown links like [link](url)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)

    # Remove @mentions like @username
    text = re.sub(r"@\w+", "", text)

    #Remove age-like patterns like 25M, 30F (common in social media data).
    text = re.sub(r'\b\d{1,3}[MF]\b', '', text)

    # Fix: Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Fix: Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Fix: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Fix: Remove newlines properly
    text = text.replace("\n", " ")

    # Fix: Remove words containing numbers
    text = re.sub(r'\b\w*\d\w*\b', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip()

    return text


def remove_stopwords_simple(text):
    if not isinstance(text, str):
        return ""

    # Tokenize without NLTK
    words = simple_tokenize(text.lower())

    # Remove stopwords
    filtered_words = [word for word in words if word not in custom_stopwords]

    return " ".join(filtered_words)

# Prediction function
def predict_sentiment(text, model):
    processed_text = preprocess_text(text)
    processed_text = remove_stopwords_simple(processed_text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction = model.predict(padded_sequence)[0]
    classes = ['anxious', 'depressed', 'happy', 'neutral']  # Example classes
    top_class_index = np.argmax(prediction)
    top_class = classes[top_class_index]
    confidence = prediction[top_class_index]
    return top_class, confidence

# Streamlit app
st.title("Sentiment Prediction App")

user_input = st.text_area("Enter your sentiment text:")
model_choice = st.selectbox("Select Model:", list(models.keys()))

if st.button("Predict"):
    model = models[model_choice]
    predicted_class, confidence = predict_sentiment(user_input, model)
    st.success(f"Predicted Sentiment: {predicted_class} ({confidence:.2f} confidence)")
