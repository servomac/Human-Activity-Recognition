FROM python:3.8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	build-essential \
	python3-dev

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
