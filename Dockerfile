FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    build-essential \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir scipy==1.11.4
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt_tab resource
RUN python -m nltk.downloader punkt_tab

# Copy application code
COPY . .

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
