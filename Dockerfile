FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn laptop_price.api:app --host ${API_HOST:-0.0.0.0} --port ${PORT:-8000}"]
