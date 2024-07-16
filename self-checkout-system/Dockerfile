# shelf-checkout/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure the YOLO config files are in the right place
RUN mkdir -p config
COPY config/yolo.cfg config/coco.names config/

# Set environment variables
ENV FLASK_APP=src/main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the application
CMD ["flask", "run"]