import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


nltk.download('stopwords')


def clean_text(text):

    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(clean_words)  # Join back into a string for TF-IDF



print("Loading data...")
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Preprocessing with NLTK ---
print("Cleaning text with NLTK (this might take a moment)...")
df['clean_message'] = df['message'].apply(clean_text)

# --- TF-IDF Vectorization ---
# TF-IDF is better than CountVectorizer because it lowers the weight of common words
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['clean_message'])
y = df['label_num']

# --- Split & Train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# --- Evaluation (Precision, Recall, Confusion Matrix) ---
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report (Precision/Recall):\n", classification_report(y_test, y_pred))

# --- Save Model & Vectorizer ---

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
print("\nSuccess! Model and Vectorizer saved.")