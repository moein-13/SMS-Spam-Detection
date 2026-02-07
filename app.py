import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string


try:
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found! Please run 'model_trainer.py' first.")
    st.stop()


def clean_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


st.markdown("""
    <style>
    /* Main background color */
    .stApp {
        background-color: #0a192f; /* Deep Navy Blue */
        color: #e6f1ff; /* Light Cyan/White text */
    }

    /* Input text area styling */
    .stTextArea textarea {
        background-color: #112240;
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #64ffda; /* Neon Green/Cyan border */
    }

    /* Button styling */
    .stButton>button {
        background-color: #64ffda;
        color: #0a192f;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4adbc4;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ccd6f6;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("📩 SMS Spam Detector")
st.markdown("Use this tool to verify if a message is **Safe (Ham)** or **Spam**.")

input_sms = st.text_area("Enter the message here", height=150,
                         placeholder="e.g. You have won a $1000 prize! Click here...")

if st.button('Analyze Message'):
    if input_sms:
        # 1. Preprocess
        transformed_sms = clean_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display Result
        if result == 1:
            st.error("🚨 SPAM DETECTED! Be careful.")
        else:
            st.success("✅ This message looks SAFE (Ham).")
    else:
        st.warning("Please enter a message to analyze.")