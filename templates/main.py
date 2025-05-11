import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Label the data
fake_df['label'] = 0  # fake
true_df['label'] = 1  # real

# Combine and shuffle
df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
