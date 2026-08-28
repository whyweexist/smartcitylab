FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./backend
# Data dirs expected via volume
RUN mkdir -p /app/data/uploads /app/data/generated
ENV DATABASE_URL=sqlite:////app/data/app.db
ENV UPLOAD_DIR=/app/data/uploads
ENV MODEL_PATH=/app/backend/artifacts/model.joblib
ENV PYTHONPATH=/app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
