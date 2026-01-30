import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure relative paths for portability
os.environ['MLFLOW_ARTIFACT_ROOT'] = "./mlruns"
mlflow.set_experiment("Reddit_Sentiment_Analysis_PRO")

def train_model():
    # 1. Load Real World Data (Kaggle-sourced Sentiment Dataset)
    # Using a subset of 50k rows for optimal performance on free-tier EC2
    DATA_URL = "https://raw.githubusercontent.com/skandavivek82/Reddit-Sentiment-Analysis/main/reddit_sentiment_data.csv"
    
    print("⏳ Downloading and loading 50,000+ rows of data...")
    try:
        df = pd.read_csv(DATA_URL).dropna()
        # Clean up column names if they vary
        df.columns = ['text', 'label']
        # Map labels if needed (Ensuring 0=Friendly, 1=Toxic)
        df['label'] = df['label'].map({-1: 1, 0: 0, 1: 0}) 
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # 2. Split data: 80% to learn, 20% to test
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # 3. Heavy Duty Pipeline
        # ngram_range(1,2) lets it see "not good" vs "good"
        # max_features keeps the model file small enough to push to Git
        pipe = Pipeline([
            ('vec', TfidfVectorizer(ngram_range=(1, 2), max_features=10000)), 
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
        print("🚀 Training the model on real-world patterns...")
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
