from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import mysql.connector
from src.model_pipeline import train_model, prepare_data, load_data_from_mysql
from sklearn.metrics import accuracy_score
import os
import logging
# Testing GitHub Actions with secrets
# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.app = FastAPI()
        self._setup_cors()
        self._load_model()
        self._define_routes()

    def _setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _load_model(self):
        try:
            model_data = joblib.load("model.joblib")
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.encoders = model_data["encoders"]
            logger.info(f"Loaded encoders: {list(self.encoders.keys())}")
        except (FileNotFoundError, KeyError, ValueError, EOFError) as e:
            self.model, self.scaler, self.encoders = None, None, None
            logger.warning(f"Warning: model.joblib not found or corrupted. Error: {str(e)}")

    def _get_mysql_connection(self):
        host = os.getenv("MYSQL_HOST", "localhost" if not os.getenv("DOCKER_ENV") else "mysql")
        return mysql.connector.connect(
            host=host,
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", "root"),
            database=os.getenv("MYSQL_DB", "churn_db"),
        )

    def _load_data_from_mysql(self):
        """Load data from MySQL (class method)."""
        try:
            conn = self._get_mysql_connection()
            df = pd.read_sql("SELECT * FROM churn_data", conn)
            conn.close()
            logger.info("\n✅ Data loaded from MySQL successfully for retraining")
            logger.info(f"MySQL columns: {df.columns.tolist()}")
            return df
        except mysql.connector.Error as err:
            logger.error(f"MySQL connection error: {err}")
            return None

    def _define_routes(self):
        @self.app.post("/predict")
        async def predict(data: dict, true_label: int = None):
            if self.model is None:
                raise HTTPException(status_code=500, detail="Model not loaded. Run training and save first.")

            if any(v is None for v in data.values()):
                raise HTTPException(status_code=400, detail="One or more fields have None values. Please check your input.")

            expected_keys = set(self.encoders.keys())
            incoming_keys = set(data.keys())
            missing_keys = expected_keys - incoming_keys
            if missing_keys:
                raise HTTPException(status_code=400, detail=f"Missing required fields: {missing_keys}")

            try:
                transformed = [self.encoders[col].transform([data[col]])[0] for col in self.encoders.keys()]
                numeric = [float(data[k]) for k in data.keys() if k not in self.encoders.keys()]
                input_data = self.scaler.transform([transformed + numeric])
                prediction = self.model.predict(input_data)[0]
                self._save_prediction(data, prediction, true_label)
                return {"prediction": int(prediction)}
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

        @self.app.post("/retrain")
        async def retrain(authorization: str = Header(None)):
            # Simple token-based authentication
            expected_token = os.getenv("RETRAIN_TOKEN", "your-secret-token")
            if authorization != f"Bearer {expected_token}":
                logger.warning("Unauthorized retrain attempt")
                raise HTTPException(status_code=403, detail="Invalid or missing authorization token")

            try:
                # Load data for retraining
                data = self._load_data_from_mysql()
                if data is None or data.empty:
                    logger.error("No data available for retraining")
                    raise HTTPException(status_code=400, detail="No data available for retraining")

                # Retrain the model
                X_train, _, y_train, _, new_scaler, new_encoders = prepare_data(data)
                new_model = train_model(X_train, y_train)

                # Save the new model
                joblib.dump({"model": new_model, "scaler": new_scaler, "encoders": new_encoders}, "model.joblib")

                # Update the in-memory model, scaler, and encoders
                self.model = new_model
                self.scaler = new_scaler
                self.encoders = new_encoders

                logger.info(f"Model retrained and reloaded successfully. New encoders: {list(self.encoders.keys())}")
                return {"status": "Model retrained and reloaded"}
            except Exception as e:
                logger.error(f"Retraining failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

        @self.app.get("/drift")
        async def get_drift():
            drift_metrics = self._compute_drift()
            if drift_metrics:
                return drift_metrics
            return {"message": "Not enough labeled data to compute drift"}

        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}

    def _save_prediction(self, data, prediction, true_label=None):
        with self._get_mysql_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """CREATE TABLE IF NOT EXISTS predictions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        data TEXT,
                        prediction INT,
                        true_label INT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
                cursor.execute(
                    "INSERT INTO predictions (data, prediction, true_label) VALUES (%s, %s, %s)",
                    (str(data), int(prediction), int(true_label) if true_label is not None else None),
                )
            conn.commit()

    def _compute_drift(self):
        with self._get_mysql_connection() as conn:
            df = pd.read_sql(
                "SELECT prediction, true_label FROM predictions WHERE true_label IS NOT NULL", conn
            )
        if len(df) > 0:
            current_accuracy = accuracy_score(df["true_label"], df["prediction"])
            drift = abs(0.85 - current_accuracy)  # BASELINE_ACCURACY = 0.85
            return {"current_accuracy": current_accuracy, "drift": drift}
        return None

# Lancer l'application
ml_service = MLService()
app = ml_service.app