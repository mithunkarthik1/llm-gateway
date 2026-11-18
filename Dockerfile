FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir poetry

COPY pyproject.toml README.md ./
RUN poetry install --no-root --no-interaction

COPY src/ ./src/
COPY utils/ ./utils/

EXPOSE 8000
CMD ["python", "-m", "src.main"]
