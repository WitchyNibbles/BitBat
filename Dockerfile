# ---------------------------------------------------------------------------
# BitBat — multi-stage Docker build
# ---------------------------------------------------------------------------
# Stage 1: Install dependencies with Poetry
# Stage 2: Lean runtime image (no Poetry, no dev deps)
# ---------------------------------------------------------------------------

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install Poetry in the builder stage only
RUN pip install poetry==1.8.5 && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --only main

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ src/
COPY streamlit/ streamlit/
COPY scripts/ scripts/
COPY .streamlit/ .streamlit/

# Default data & model directories (mount as volumes in production)
RUN mkdir -p data models logs config

# Default config
COPY src/bitbat/config/default.yaml config/default.yaml

EXPOSE 8000 8501

# Default: run the FastAPI server
CMD ["uvicorn", "bitbat.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
