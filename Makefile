.PHONY: install test lint format


install:
	pdm install

test:
	pdm run pytest

lint:
	pdm run ruff mads_datasets
	pdm run mypy mads_datasets

format:
	pdm run isort -v mads_datasets
	pdm run black mads_datasets