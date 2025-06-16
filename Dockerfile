# Dockerfile for Scouting MVP Flask App

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]