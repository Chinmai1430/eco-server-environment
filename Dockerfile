FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn pydantic

# Copy all project files
COPY . .

# Expose the port OpenEnv expects
EXPOSE 8080

# Start the FastAPI inference server
CMD ["python", "inference.py", "--mode", "server", "--host", "0.0.0.0", "--port", "8080"]
