FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn pydantic requests

COPY . .

EXPOSE 7860

CMD ["python", "inference.py", "--host", "0.0.0.0", "--port", "7860"]
