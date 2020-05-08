"""Dockerfile for deploy on GCP."""

FROM tensorflow/tensorflow:latest-gpu-py3

COPY requirements.txt /tmp/pip-tmp/

RUN apt update && apt install -y unzip


RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp


ENV PYTHONPATH="/workspaces/DD2424-project"

CMD pre-commit install
