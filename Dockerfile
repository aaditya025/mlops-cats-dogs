# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for cv2 or other libs sometimes, but minimal here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# Copy requirements first to leverage cache
COPY requirements.txt .

# Remove dvc from requirements for docker build to reduce size/complexity if not needed at runtime,
# or keep it. DVC might need git.
# Modify requirements line to remove version pin for tensorflow matching the one we used
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY src/ src/
COPY models/ models/
COPY .dvc/ .dvc/
COPY dvc.yaml dvc.yaml
# We copy models directly for now. In a real CD, we might pull from DVC/S3.

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
