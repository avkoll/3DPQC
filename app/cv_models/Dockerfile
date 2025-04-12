FROM python:3.12-slim

WORKDIR /cv_models

RUN pip install --no-cache-dir opencv-python numpy

COPY . .

CMD ["python", "stringing.py"]
