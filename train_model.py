import joblib
import numpy as np
from app.utils.preprocess import clean_text

def load_model_and_vectorizer():
    model = joblib.load('model/fake_news_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
    return model, vectorizer

def predict_news(text, model, vectorizer):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    prob = np.max(model.predict_proba(vector)) * 100
    return pred, round(prob, 2)
