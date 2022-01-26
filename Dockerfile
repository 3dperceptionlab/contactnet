FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install -y gcc git wget

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt