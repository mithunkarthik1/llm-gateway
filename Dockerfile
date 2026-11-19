FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir poetry

COPY pyproject.toml README.md alembic.ini* ./
RUN poetry install --no-root --no-interaction

COPY src/ ./src/
COPY utils/ ./utils/
COPY migrations/ ./migrations/

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
