FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /workspaces/DD2424-project

COPY requirements.txt /tmp/pip-tmp/

RUN apt update && apt install -y unzip
RUN apt install -y git
RUN pip3 install pre-commit

COPY . .

ENV PYTHONPATH="/workspaces/DD2424-project"

CMD pre-commit install
