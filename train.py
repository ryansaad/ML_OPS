import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.environ['MLFLOW_ARTIFACT_ROOT'] = "./mlruns"
mlflow.set_experiment("Reddit_Sentiment_Analysis_PRO")

def train_model():
    print("⏳ Loading dataset...")
    # Using 'sci.space' and 'alt.atheism' as proxies for distinct language patterns
    categories = ['sci.space', 'alt.atheism']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    df = pd.DataFrame({'text': newsgroups.data, 'label': newsgroups.target})
    df = df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    with mlflow.start_run():
        # Added stop_words and increased max_features for better nuance
        pipe = Pipeline([
            ('vec', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words='english')), 
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')) # Balanced helps with bias
        ])
        
        print("🚀 Training smarter model...")
        pipe.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, pipe.predict(X_test))
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(sk_model=pipe, artifact_path="sentiment_model")
        
        print(f"🎯 Model Accuracy: {int(acc * 100)}")

if __name__ == "__main__":
    train_model()
