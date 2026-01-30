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
mlflow.set_experiment("Reddit_Sentiment_Analysis")

def train_model():
    # 1. Expanded dataset to provide more patterns
    data = {
        'text': [
            "I love this", "Great!", "Amazing", "Friendly", "So helpful", "Kind words",
            "I hate this", "Toxic", "Bad", "Terrible", "Useless", "Stay away",
            "This is wonderful", "You are the best", "Disgusting", "Horrible service"
        ],
        'label': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1] 
    }
    df = pd.DataFrame(data)

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=42
    )

    with mlflow.start_run() as run:
        # 3. Define Pipeline (Using Bigrams to catch more context)
        pipe = Pipeline([
            ('vec', TfidfVectorizer(ngram_range=(1, 2))), 
            ('clf', LogisticRegression())
        ])
        pipe.fit(X_train, y_train)

        # 4. Evaluate
        predictions = pipe.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        
        # 5. Log results to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(sk_model=pipe, artifact_path="sentiment_model")
        
        print("\n--- Model Performance Report ---")
        print(f"🎯 Accuracy: {acc * 100}%")
        print("\n📊 Confusion Matrix:")
        print("Format: [[True_Friendly, False_Toxic],")
        print("         [False_Friendly, True_Toxic]]")
        print(cm)
        print(f"\n✅ NEW RUN ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
