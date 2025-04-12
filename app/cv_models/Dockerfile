FROM python:3.12-slim

WORKDIR /cv_models

# Update and install the required system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy the local requirements file
COPY cv_requirements.txt .

RUN pip install --no-cache-dir -r cv_requirements.txt

COPY . .

CMD ["python", "stringing.py"]
