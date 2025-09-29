install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black gold_analysis.py

lint:
	flake8 --ignore=C,N gold_analysis.py

test:
	python -m pytest -vv --cov=hello test_hello.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test
