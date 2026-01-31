.PHONY: fmt lint test streamlit

fmt:
	poetry run ruff format src tests
	poetry run black src tests

lint:
	poetry run ruff check src tests
	poetry run mypy src tests

test:
	poetry run pytest

streamlit:
	poetry run streamlit run streamlit/app.py
