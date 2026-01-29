import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Ensure relative paths for portability
os.environ['MLFLOW_ARTIFACT_ROOT'] = "./mlruns"
mlflow.set_experiment("Reddit_Sentiment_Analysis")

def train_model():
    # 1. Expanded (but still small) dataset for logic testing
    data = {
        'text': [
            "I love this", "Great!", "Amazing", "Friendly", 
            "I hate this", "Toxic", "Bad", "Terrible"
        ],
        'label': [0, 0, 0, 0, 1, 1, 1, 1] # 0 = Friendly, 1 = Toxic
    }
    df = pd.DataFrame(data)

    # 2. Split data: 75% to learn, 25% to be tested on
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=42
    )

    with mlflow.start_run() as run:
        # 3. Define and Fit Pipeline
        pipe = Pipeline([
            ('vec', TfidfVectorizer()), 
            ('clf', LogisticRegression())
        ])
        pipe.fit(X_train, y_train)

        # 4. Evaluate Accuracy
        predictions = pipe.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # 5. Log results
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(sk_model=pipe, artifact_path="sentiment_model")
        
        print(f"🎯 Model Accuracy: {acc * 100}%")
        print(f"✅ NEW RUN ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
