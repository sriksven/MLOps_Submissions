# Dockerfile

# 1. Use a lightweight Python image
FROM python:3.10-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the entire project into the container
COPY . .

# 5. Default command — runs the training script
CMD ["python", "src/train.py"]
