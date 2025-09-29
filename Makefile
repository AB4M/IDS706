PYTHON=python

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt

format:
	black hello.py gold_analysis.py

format-check:
	black --check hello.py gold_analysis.py

lint:
	# 忽略常见的行宽/切片空格/换行操作符；F401(未使用导入)我们已在代码里清理了
	flake8 --ignore=E203,E501,W503 hello.py gold_analysis.py

test:
	$(PYTHON) -m pytest -q

all: install format lint test
