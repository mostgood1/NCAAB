# Base lightweight Python image
FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    RENDER=1 \
    FLASK_ENV=production \
    FLASK_DEBUG=0

WORKDIR /app

# System deps (add build-essential if you later compile ORT or Torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY app.py ./
COPY src ./src
COPY templates ./templates
COPY static ./static
COPY outputs ./outputs
COPY data ./data
COPY pyproject.toml ./
COPY README.md ./

# Expose port (Render sets $PORT, but this helps local runs)
EXPOSE 5050

# Default: run single-process Flask app binding to $PORT (Render provides PORT)
CMD ["python", "app.py"]
