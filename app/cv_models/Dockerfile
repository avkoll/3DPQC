FROM python:3.12-slim
WORKDIR /cv_models

# point at where we'll mount it
ENV MODEL_PATH=/models/

# system deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# install build tools so scikit‑learn (and any other C extensions) can compile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY cv_requirements.txt .
RUN pip install --no-cache-dir -r cv_requirements.txt

# copy your inference scripts
COPY . .

CMD ["python", "stringing.py"]
