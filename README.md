# 📩 SMS Spam Detection System

## 📌 Overview
An AI-based SMS Spam Classification system built with **Python** and **Machine Learning**. 
This application analyzes text messages to detect spam using 
**NLP (Natural Language Processing)** techniques and a 
**Naive Bayes** classifier. 
It features a modern web interface deployed using **Streamlit**.

## 🚀 Features
* **Real-time Prediction:** Instantly classifies messages as "Spam" or "Ham" (Safe).
* **High Accuracy:** achieved **97% accuracy** with **100% Precision** on the test dataset.
* **GUI:** Interactive web-based user interface with a custom dark-mode design.
* **Tech Stack:** Python, Scikit-Learn, NLTK, Pandas, Streamlit.

## 🛠️ How It Works
1.  **Preprocessing:** Cleans text using NLTK (removes stopwords, punctuation).
2.  **Vectorization:** Converts text to numbers using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
3.  **Modeling:** Uses **Multinomial Naive Bayes** for probabilistic classification.


## 📊 Model Performance
* **Accuracy:** 97%
* **Precision (Spam):** 1.00 (0 False Positives)
* **Recall (Spam):** 0.75
