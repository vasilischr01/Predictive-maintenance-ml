install:
	pip install -r requirements.txt

data:
	python -m src.data.download

train:
	python -m src.training.train

api:
	uvicorn src.api.main:app --reload

test:
	pytest -q
