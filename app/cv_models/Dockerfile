FROM python:3.12-slim

WORKDIR /cv_models

# Copy the local requirements file
COPY cv_requirements.txt .

RUN pip install --no-cache-dir -r cv_requirements.txt

COPY . .

CMD ["python", "stringing.py"]
