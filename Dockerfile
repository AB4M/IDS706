# Use slim Python base image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (minimal; wheels cover most)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Working directory matches paths used by tests
WORKDIR /mnt/data

# Install Python dependencies first for better layer caching
COPY requirements.txt /mnt/data/requirements.txt
RUN pip install --upgrade pip && pip install -r /mnt/data/requirements.txt

# Copy your source and tests
COPY gold_analysis.py /mnt/data/gold_analysis.py
COPY tests /mnt/data/tests

# Default command: run tests quietly
CMD ["pytest", "-q", "tests"]
