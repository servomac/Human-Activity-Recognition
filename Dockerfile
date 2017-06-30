FROM python:3.6

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	build-essential \
	python3-dev

RUN pip install \
	https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.0-cp36-cp36m-linux_x86_64.whl

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
