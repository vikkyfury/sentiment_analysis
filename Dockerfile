FROM python:3.9-slim
WORKDIR /app

# Install only the Python packages you need
COPY requirements.txt .
RUN pip install --no-cache-dir flask mlflow scikit-learn pandas

# Copy in your application code AND the vendored model files
COPY . .

EXPOSE 5000
CMD ["python", "src/serve.py"]

