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
    # 1. Using a verified, stable Sentiment Dataset URL
    DATA_URL = "https://raw.githubusercontent.com/skandavivek82/Reddit-Sentiment-Analysis/main/reddit_sentiment_data.csv"
    
    print("⏳ Downloading and loading real-world data...")
    try:
        # Some CSVs use different encodings; 'latin-1' is a safe bet for social media text
        df = pd.read_csv(DATA_URL, encoding='utf-8').dropna()
        
        # Ensure we have the right columns (adjusting for the specific dataset structure)
        # Most of these datasets have 'clean_comment' and 'category'
        if 'clean_comment' in df.columns:
            df = df.rename(columns={'clean_comment': 'text', 'category': 'label'})
        
        # Ensure labels are 0 (Friendly) and 1 (Toxic)
        # This dataset often uses -1 for negative, 0 for neutral, 1 for positive.
        # We will map -1 (Negative) to 1 (Toxic) and everything else to 0 (Friendly).
        df['label'] = df['label'].map({-1: 1, 0: 0, 1: 0})
        
        print(f"✅ Loaded {len(df)} rows successfully.")
    except Exception as e:
        # Fallback to a secondary stable link if the first fails
        print(f"⚠️ Primary URL failed, trying fallback...")
        DATA_URL_ALT = "https://raw.githubusercontent.com/shauryauppal/PySpark-Sentiment-Analysis/master/reddit_train.csv"
        df = pd.read_csv(DATA_URL_ALT).dropna()
        df.columns = ['text', 'label']

    # 2. Split data: 80% to learn, 20% to test
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # 3. High-Performance Pipeline
        # max_features=5000 keeps the model light but effective
        pipe = Pipeline([
            ('vec', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)), 
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
