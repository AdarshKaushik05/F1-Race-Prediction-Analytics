# Use a slim Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Render uses port 10000 for its web services
EXPOSE 10000

# Command to start the FastAPI "Brain"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
