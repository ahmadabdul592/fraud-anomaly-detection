.PHONY: install test run run-semi
install:
	pip install -e ".[dev]"
test:
	pytest -q
run:
	python -m fraudad.train --mode unsupervised
run-semi:
	python -m fraudad.train --mode semi
