# Use an official Python base image
FROM python:3.11-slim

# Install system dependencies (PortAudio for sounddevice)
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (use the same port as in your app)
ENV PORT=5000
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
