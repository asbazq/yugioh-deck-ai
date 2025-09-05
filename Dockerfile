FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system deps needed by opencv-python-headless and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Default model and runtime configs (can be overridden via env)
ENV MODEL_PATH=/app/best.pt \
    INPUT_SIZE=640 \
    TOP_K=5 \
    ALLOW_CORS=1

EXPOSE 8000

# Startup script: optional snapshot import then run API
RUN chmod +x scripts/start.sh
CMD ["scripts/start.sh"]
