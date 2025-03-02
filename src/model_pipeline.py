import pandas as pd
import joblib
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import os
from imblearn.over_sampling import SMOTE
from elasticsearch import Elasticsearch
import json

# Set MLflow tracking URI to the server
mlflow.set_tracking_uri("http://mlflow:5000")
os.environ.pop("MLFLOW_TRACKING_URI", None)

# Ensure the Default experiment exists
try:
    mlflow.set_experiment("Default")
except:
    mlflow.create_experiment("Default")
    mlflow.set_experiment("Default")

# Initialize Elasticsearch client
es = Elasticsearch("http://elasticsearch:9200")

def load_data_from_csv(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    print("\n✅ CSV data loaded successfully")
    print(f"CSV columns: {df.columns.tolist()}")
    return df

def upload_data_to_mysql(df):
    """Upload data from DataFrame to MySQL."""
    db_config = {
        "host": os.getenv("MYSQL_HOST", "mysql"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "root"),
        "database": os.getenv("MYSQL_DB", "churn_db"),
        "port": 3306
    }
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS churn_data")
        cursor.execute("""
            CREATE TABLE churn_data (
                State VARCHAR(2),
                Account_length INT,
                Area_code INT,
                International_plan VARCHAR(3),
                Voice_mail_plan VARCHAR(3),
                Number_vmail_messages INT,
                Total_day_minutes FLOAT,
                Total_day_calls INT,
                Total_day_charge FLOAT,
                Total_eve_minutes FLOAT,
                Total_eve_calls INT,
                Total_eve_charge FLOAT,
                Total_night_minutes FLOAT,
                Total_night_calls INT,
                Total_night_charge FLOAT,
                Total_intl_minutes FLOAT,
                Total_intl_calls INT,
                Total_intl_charge FLOAT,
                Customer_service_calls INT,
                Churn INT
            )
        """)
        expected_columns = [
            "State", "Account_length", "Area_code", "International_plan",
            "Voice_mail_plan", "Number_vmail_messages", "Total_day_minutes",
            "Total_day_calls", "Total_day_charge", "Total_eve_minutes",
            "Total_eve_calls", "Total_eve_charge", "Total_night_minutes",
            "Total_night_calls", "Total_night_charge", "Total_intl_minutes",
            "Total_intl_calls", "Total_intl_charge", "Customer_service_calls",
            "Churn"
        ]
        print(f"CSV columns: {df.columns.tolist()}")
        if list(df.columns) != expected_columns:
            print(f"Warning: CSV columns {list(df.columns)} do not match expected {expected_columns}")
            df = df.reindex(columns=expected_columns, fill_value=0)

        df = df.dropna()
        print(f"After dropping nulls, {len(df)} rows remain")
        print(f"Churn value counts in CSV: {df['Churn'].value_counts().to_dict()}")
        df["Churn"] = df["Churn"].apply(lambda x: 1 if str(x).strip().upper() == "TRUE" else 0)

        inserted_rows = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO churn_data (
                        State, Account_length, Area_code, International_plan,
                        Voice_mail_plan, Number_vmail_messages, Total_day_minutes,
                        Total_day_calls, Total_day_charge, Total_eve_minutes,
                        Total_eve_calls, Total_eve_charge, Total_night_minutes,
                        Total_night_calls, Total_night_charge, Total_intl_minutes,
                        Total_intl_calls, Total_intl_charge, Customer_service_calls,
                        Churn
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, tuple(row))
                inserted_rows += 1
            except mysql.connector.Error as err:
                print(f"Row insertion failed: {err}")
                print(f"Row data: {row.tolist()}")
                raise
        conn.commit()
        print(f"Data uploaded to MySQL successfully. Inserted {inserted_rows} rows.")
    except mysql.connector.Error as err:
        print(f"MySQL error: {err}")
    finally:
        if conn and conn.is_connected():
            conn.close()

def load_data_from_mysql():
    """Load data from MySQL."""
    db_config = {
        "host": os.getenv("MYSQL_HOST", "mysql"),
        "user": "root",
        "password": "root",
        "database": "churn_db",
        "port": 3306,
    }
    try:
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql("SELECT * FROM churn_data", conn)
        conn.close()
        print("\n✅ Data loaded from MySQL successfully")
        print(f"MySQL columns: {df.columns.tolist()}")
        return df
    except mysql.connector.Error as err:
        print(f"MySQL connection error: {err}")
        return None

def prepare_data(df):
    """Prepare data for training."""
    print(f"Raw Churn values: {df['Churn'].value_counts().to_dict()}")
    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == 1 else 0)
    print(f"Converted Churn distribution: {df['Churn'].value_counts().to_dict()}")

    categorical_columns = [
        "State",
        "International_plan",
        "Voice_mail_plan",
    ]
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            print(f"Warning: Column {col} not found in DataFrame. Available columns: {df.columns.tolist()}")

    features = df.drop(columns=["Churn"])
    target = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    print(f"Before SMOTE - Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    print(f"Before SMOTE - Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Before SMOTE - Test class distribution: {y_test.value_counts().to_dict()}")

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Train set size: {len(X_train)}")
    print(f"After SMOTE - Train class distribution: {y_train.value_counts().to_dict()}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, label_encoders

def train_model(X_train, y_train):
    """Train an MLP neural network with MLflow tracking."""
    with mlflow.start_run():
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=42,
        )
        model.fit(X_train, y_train)
        mlflow.log_param("hidden_layer_sizes", (100,))
        mlflow.log_param("activation", "relu")
        mlflow.log_param("solver", "adam")
        mlflow.log_param("max_iter", 200)
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", "ChurnModel")
        print("\n✅ Model trained successfully")
        return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set and log to Elasticsearch."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Test set predictions (first 10): {y_pred[:10]}")
    print(f"Test set true labels (first 10): {y_test[:10]}")
    print(f"Test set accuracy: {accuracy}")
    print(f"Classification report:\n{report}")
    with mlflow.start_run(nested=True):
        mlflow.log_metric("test_accuracy", accuracy)

    # Log predictions to Elasticsearch
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        doc = {
            "prediction": int(pred),
            "true_label": int(true),
            "timestamp": pd.Timestamp.now().isoformat(),
            "sample_id": i
        }
        try:
            es.index(index="churn_predictions", body=doc)
        except Exception as e:
            print(f"Failed to log prediction to Elasticsearch: {e}")

    return accuracy, report

def save_model(model, scaler, label_encoders, file_path):
    """Save the model, scaler, and label encoders."""
    model_dict = {
        "model": model,
        "scaler": scaler,
        "encoders": label_encoders
    }
    joblib.dump(model_dict, file_path)

def load_model(file_path):
    """Load the model and associated objects."""
    model_dict = joblib.load(file_path)
    return model_dict["model"]