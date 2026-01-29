import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import os
os.environ['MLFLOW_ARTIFACT_ROOT'] = "./mlruns"
mlflow.set_experiment("Reddit_Sentiment_Analysis")

with mlflow.start_run() as run:
    df = pd.DataFrame({
        'text': ["I love this", "Great", "I hate this", "Toxic"],
        'label': [0, 0, 1, 1]
    })
    pipe = Pipeline([('vec', TfidfVectorizer()), ('clf', LogisticRegression())])
    pipe.fit(df['text'], df['label'])
    mlflow.sklearn.log_model(sk_model=pipe, artifact_path="sentiment_model")
    print(f"✅ NEW RUN ID: {run.info.run_id}")
