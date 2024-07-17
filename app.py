import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the trained model
model = joblib.load('books.pkl')

# Define functions for preprocessing and TF-IDF transformation
def lemmatize_stopwords(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def tfidf_transform(text):
    tfidf_vectorizer = TfidfVectorizer(max_features=150000)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    return tfidf_matrix

# Streamlit app
st.title("A Book Genre Classification APP")

text_input = st.text_area('Please Enter the summary\\text of your book ')

if st.button('Predict'):
    processed_text = lemmatize_stopwords(text_input)
    tfidf_matrix = tfidf_transform(processed_text)
    prediction = model.predict(tfidf_matrix)
    
    mapping = {
        0: 'Fantasy',
        1: 'Science Fiction',
        2: 'Crime Fiction',
        3: 'Historic Novel',
        4: 'Horror',
        5: 'Thriller'
    }
    
    prediction_label = mapping.get(prediction[0], "Unknown")
    st.write("Predicted Genre:", prediction_label)
