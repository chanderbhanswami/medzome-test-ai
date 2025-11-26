FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (libgl1 replaces obsolete libgl1-mesa-glx)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the server
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1"]
