FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install curl and wget for healthcheck
RUN apt-get update && apt-get install -y curl wget && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY data/ ./data/
COPY model.joblib .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]