FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment setup (these will be overridden by Cloud Run env vars)
ENV DEBUG=False
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Run the application
CMD exec uvicorn app.main:app --host ${API_HOST} --port ${API_PORT} --workers 1