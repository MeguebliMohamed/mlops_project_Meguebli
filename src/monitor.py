import psutil
from elasticsearch import Elasticsearch
import mysql.connector
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import os

es = Elasticsearch("http://elasticsearch:9200")
BASELINE_ACCURACY = 0.85


def compute_drift():
    conn = mysql.connector.connect(
        host="mysql",
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
        database=os.getenv("MYSQL_DB", "churn_db"),
    )
    df = pd.read_sql(
        "SELECT prediction, true_label FROM predictions "
        "WHERE true_label IS NOT NULL",
        conn,
    )
    conn.close()
    if len(df) > 0:
        current_accuracy = accuracy_score(df["true_label"], df["prediction"])
        drift = abs(BASELINE_ACCURACY - current_accuracy)
        return {"current_accuracy": current_accuracy, "drift": drift}
    return None


while True:
    system_metrics = {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
    }
    es.index(index="system_metrics", body=system_metrics)
    drift_metrics = compute_drift()
    if drift_metrics:
        es.index(index="model_drift", body=drift_metrics)
        if drift_metrics["drift"] > 0.1:
            print(
                f"⚠️ Drift detected: {drift_metrics['drift']}. Consider \
                   retraining."
            )
    time.sleep(60)
