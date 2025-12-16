# Use Python 3.10
FROM python:3.10-slim

# Set working folder
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Command to run when the container starts
CMD ["python", "compare_models.py"]