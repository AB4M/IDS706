PYTHON=python

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

format:
	black hello.py gold_analysis.py

format-check:
	black --check hello.py gold_analysis.py

lint:
	flake8 --ignore=E203,E501,W503 hello.py gold_analysis.py

test:
	$(PYTHON) -m pytest -q

all: install format lint test
