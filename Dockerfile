# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements/base.txt requirements/base.txt
COPY requirements/ml.txt requirements/ml.txt
COPY requirements/dev.txt requirements/dev.txt

# Install Python dependencies
# Base requirements (always installed)
RUN pip install --no-cache-dir -r requirements/base.txt

# ML requirements (optional, comment out if not needed)
RUN pip install --no-cache-dir -r requirements/ml.txt || echo "Some ML dependencies could not be installed"

# Dev requirements (optional, comment out in production)
# RUN pip install --no-cache-dir -r requirements/dev.txt

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create a non-root user to run the application
RUN useradd -m hyperliquid
USER hyperliquid

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run when container starts
ENTRYPOINT ["python", "-m", "hyperliquidmaster.scripts.master_bot"]
CMD ["--config", "config.json"]
