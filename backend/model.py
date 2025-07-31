import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

def train():
    df = pd.read_csv('train_emotions.csv', delimiter=';', header=None, names=['content', 'label'])
    print("Columns:", df.columns)
    X = df['content']
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_tfidf, y_train)
    pickle.dump({'tfidf': tfidf, 'model': model}, open('emotion_model.pkl', 'wb'))

def predict(text):
    data = pickle.load(open('emotion_model.pkl', 'rb'))
    clean = text.lower()
    vec = data['tfidf'].transform([clean])
    pred = data['model'].predict(vec)[0]
    return pred
