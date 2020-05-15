FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

WORKDIR /workspaces/DD2424-project

COPY requirements.txt /tmp/pip-tmp/

RUN apt update
RUN apt install -y unzip
RUN apt install -y git
RUN pip3 install pre-commit

COPY . .

ENV PYTHONPATH="/workspaces/DD2424-project"

CMD jupyter notebook --host=0.0.0.0 --port=8888 --allow-root
