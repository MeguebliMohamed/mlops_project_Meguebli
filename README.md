# MLOps Project
A complete MLOps pipeline for churn prediction with model drift detection.

## Setup
1. Place your `data.csv` in the `data/` directory.
2. Install dependencies: `make setup`
3. Start containers: `make docker-up`
4. Run the full pipeline: `make all`
5. Access FastAPI at `http://localhost:8000/docs`
6. View MLflow at `http://localhost:5000`
7. Monitor with Kibana at `http://localhost:5601`

## Commands
- Upload CSV to MySQL: `make upload`
- Train model: `make train`
- Evaluate: `make evaluate`
- Save model: `make save`
- Run tests: `make test`
- Monitor system and drift: `make monitor`
- Check drift manually: `make drift`

## Drift Detection
- Model performance drift is monitored by comparing current accuracy (from predictions with true labels) to a baseline (set to 0.85).
- Drift metrics are logged to Elasticsearch (`model_drift` index) and can be visualized in Kibana.
- Provide `true_label` in `/predict` requests to enable drift calculation.
