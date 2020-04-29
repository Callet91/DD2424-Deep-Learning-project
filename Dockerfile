FROM tensorflow/serving:latest-devel-gpu
USER root
WORKDIR /tensorflow/DD2424

COPY . /tensorflow/DD2424

RUN python3.7 -m pip install pre-commit
