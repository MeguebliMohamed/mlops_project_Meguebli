PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
MODEL_PATH=model.joblib
CSV_PATH=data/data.csv

.PHONY: setup
setup:
	@$(PYTHON) -m venv $(ENV_NAME)
	@. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)

.PHONY: docker-build docker-up docker-down
docker-build:
	@docker-compose build
docker-up:
	@docker-compose up -d
docker-down:
	@docker-compose down

.PHONY: lint format security
lint:
	@. $(ENV_NAME)/bin/activate && flake8 src/
format:
	@. $(ENV_NAME)/bin/activate && black --line-length 79 src/
security:
	@. $(ENV_NAME)/bin/activate && bandit -r src/

.PHONY: test_unit test_pipeline
test_unit:
	@echo "🧪 Exécution des tests unitaires..."
	@. $(ENV_NAME)/bin/activate && PYTHONPATH=. pytest tests/test_pipeline.py -k "test_load_data or test_prepare_data or test_train_model"

test_pipeline:
	@echo "🛠 Exécution des tests fonctionnels..."
	@. $(ENV_NAME)/bin/activate && PYTHONPATH=. pytest tests/test_pipeline.py -k "test_evaluate_model"

.PHONY: upload train save evaluate full_pipeline
upload:
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m src.main upload --csv $(CSV_PATH)
train:
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m src.main train
save:
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m src.main save --save $(MODEL_PATH)
evaluate:
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m src.main evaluate
full_pipeline:
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m src.main full_pipeline --csv $(CSV_PATH) --save $(MODEL_PATH)

.PHONY: all
all: setup docker-build docker-up test_unit test_pipeline full_pipeline
	@echo "🎉 Project fully executed!"

.PHONY: monitor
monitor:
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m src.monitor &

.PHONY: drift
drift:
	@curl http://localhost:8000/drift