import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure relative paths for portability
os.environ['MLFLOW_ARTIFACT_ROOT'] = "./mlruns"
mlflow.set_experiment("Reddit_Sentiment_Analysis_PRO")

def train_model():
    print("⏳ Loading professional-grade dataset (20 Newsgroups)...")
    
    # 1. Load data from scikit-learn (Reliable, no 404 risks)
    # We'll take two distinct categories to simulate "Friendly" vs "Toxic" logic
    categories = ['sci.space', 'alt.atheism']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    
    df = pd.DataFrame({'text': newsgroups.data, 'label': newsgroups.target})
    df = df.dropna()
    
    print(f"✅ Loaded {len(df)} rows successfully.")

    # 2. Split data: 80% to learn, 20% to test
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # 3. High-Performance Pipeline
        # ngram_range(1,2) lets it see context; max_features keeps the model file small
        pipe = Pipeline([
            ('vec', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')), 
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
        print("🚀 Training the model on real-world language patterns...")
        pipe.fit(X_train, y_train)

        # 4. Evaluation
        predictions = pipe.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        # 5. Log results
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(sk_model=pipe, artifact_path="sentiment_model")
        
        print("\n--- Model Performance Report ---")
        print(f"🎯 Model Accuracy: {int(acc * 100)}")
        print("\n📊 Confusion Matrix:")
        print(cm)
        print(f"\n✅ NEW RUN ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
