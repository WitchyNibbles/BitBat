.PHONY: fmt lint test

fmt:
	poetry run ruff format src tests
	poetry run black src tests

lint:
	poetry run ruff check src tests
	poetry run mypy src tests

test:
	poetry run pytest
