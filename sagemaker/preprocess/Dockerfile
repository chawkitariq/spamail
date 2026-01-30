FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/processing/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess.py /opt/ml/processing/code/preprocess.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/opt/ml/processing/code/preprocess.py"]