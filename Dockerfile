FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/
COPY data/ ./data/

RUN pip install --no-cache-dir .
RUN python -m laptop_price.train --no-tuning

EXPOSE 8000

CMD ["sh", "-c", "uvicorn laptop_price.api:app --host ${API_HOST:-0.0.0.0} --port ${PORT:-8000}"]
