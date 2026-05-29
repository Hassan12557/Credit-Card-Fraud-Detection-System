# Use a lightweight optimized Python base runtime layer
FROM python:3.10-slim

# Prevent Python from writing .pyc files to disk and ensure direct output stream logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish isolated internal application working scope directory
WORKDIR /app

# Install native OS dependencies required for basic compiling elements if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency layer structures to optimize rebuilt times
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Mirror complete project repository structure across container filesystem boundaries
COPY . /app/

# Open internal virtual network access pathways to designated endpoint port
EXPOSE 5000

# Fire production service instance gateway worker on startup
CMD ["python", "API/app.py"]