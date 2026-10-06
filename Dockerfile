FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Default: run the offline baseline evaluation (no API key needed).
CMD ["python", "-m", "cne.cli", "eval", "--extractor", "baseline"]
