# Use a lightweight, official Python image for a smaller container size
FROM python:3.9-slim

# Set a dedicated working directory for better organization
WORKDIR /app

# Copy and install requirements first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container
COPY . .

# Inform Docker that the container listens on port 8000
EXPOSE 8000

# Run the app using Gunicorn, a production-grade web server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```
