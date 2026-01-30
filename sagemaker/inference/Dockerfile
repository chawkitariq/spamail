FROM python:3.9-slim

WORKDIR /opt/program

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py /opt/program/
COPY serve /opt/program/
COPY wsgi.py /opt/program/
COPY nginx.conf /opt/program/

RUN chmod +x /opt/program/serve

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

EXPOSE 8080

ENTRYPOINT ["serve"]
