# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the dependency files and install Python dependencies
COPY requirements.txt ./
# Install only the necessary dependencies (Flask, google-genai, gunicorn)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY main.py .

# Expose the port that the service will run on
ENV PORT 8080
EXPOSE 8080

# Run gunicorn as the production server
# The command runs the Flask app 'app' found in 'main.py'
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app