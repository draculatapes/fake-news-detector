from flask import Flask, render_template, request
from app.model import load_model_and_vectorizer, predict_news

app = Flask(__name__)

# Load model and vectorizer once when the app starts
model, vectorizer = load_model_and_vectorizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    prediction, confidence = predict_news(news_text, model, vectorizer)
    result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
    return render_template('index.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
