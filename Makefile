PYTHON=python

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

format:
	# Auto-format in place
	black hello.py gold_analysis.py

format-check:
	# Fail CI if formatting is needed
	black --check hello.py gold_analysis.py

lint:
	flake8 --ignore=E203,E501,W503 hello.py gold_analysis.py

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf __pycache__ .pytest_cache .coverage

all: install format lint test
