.PHONY: install test lint format


install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run flake8 mads_datasets
	poetry run mypy mads_datasets

format:
	poetry run isort -v mads_datasets
	poetry run black mads_datasets