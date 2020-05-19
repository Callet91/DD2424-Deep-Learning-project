# DD2424-project: Training Convolutional Networks for image classification using AlexNet architecture
By: Carl Jensen, Jacob Röing, Almida de Val and Alejandro Bergstrand

## Project description
For the project assignment in the course DD2424, the group wants to go with one of the recommended sample projects, namely the Explore training ConvNets from scratch for image classification project. The motivation for this is that the group has no previous knowledge, outside the scope of this course, with Deep Learning or any similar subject. Therefore, the group wanted to explore and learn more about Convolutional Networks and especially focus on creating an architecture similar to AlexNet since AlexNet was a game changer in training deep neural networks.

## Training, validation and testing data
A subset of ImageNet (tiny_imagenet) provided by the instructors of Stanford’s CS231 course.

## Software package
The code will be written in Python3 and Tensorflow and numpy as the main software package.

## Implementation
The group will build up an architecture similar to AlexNet from scratch using Tensorflow. The level of success will be measured in the system’s ability to accurately classify the objects in the pictures.

## Goals and grade
Since all of the group members have approximately the same level of coding experience, all the group members have similar goals with regards to acquiring knowledge from the project. The goal is to get a deeper understanding of Convolutional neural networks and the architecture of AlexNet. Furthermore, the members will learn how to implement training using Tensorflow and python as a programming language. As a group projects, each member will also get more experience in coding the system together.

The group is aiming for the grade C.

## Initial experiments
The initial goal is to successfully set up AlexNet in classifying the images in the ImageNet dataset. The two initial experiments will be to create an average object for each label of the trained network and to measure the accuracy of the system.

## Measurement of success
To measure the performance of the network, the network will be trained, validated and tested on separate datasets from a subset of the ImageNet. The results will be compared with the results achieved by students from the course cs231 held at Stanford.

## References
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, 2012,
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

 ILSVRC: imagenet large scale visual recognition competition. http://www.image-net.org/ challenges/LSVRC/

# Setting up dev environment

If you like to start develop on your own computer, there are two neat options for doing so: 

## Option 1: Using Docker
- Download and install Docker on your OS https://docs.docker.com/get-docker/
- Download and install VS code https://code.visualstudio.com/
- In VS code, make sure you install the Docker exstension by opening the extensions view `(Ctrl+Shift+X)`, search for docker and select Docker extension authored by Microsoft. Also, make sure you install the Remote-container exstension when you are at it.
- Clone this repo.
```sh
cd /path/to/your/directory && git clone https://github.com/Callet91/DD2424-project.git
```
- Open the repo with VS code
```sh
cd /path/to/your/directory && code .
```
- Now, in VS code, open the remote container settings `(CTRL+SHIFT+P)` and then search for the option `Remote-Containers: Reopen in Container` and select this option.
- PS. The first time you open the container it can take some time due to that the container needs to be created from the image.
- Open a new terminal in VS code and download the tiny-imagenet-200 dataset.
```sh
mkdir /workspaces/DD2424-project/dataset && cd /workspaces/DD2424-project/dataset/ && wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip && rm tiny-imagenet-200.zip
```
## Option 2: Using venv
- Clone this repo.
```sh
cd /path/to/your/directory && git clone https://github.com/Callet91/DD2424-project.git
```
- Make sure you have installed `Python3`, `pip3` and `virtualenv`.
- If you have not everything installed run the following commands.

Ubuntu/WSL:
```sh
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv
```

MacOS:
```sh
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
export PATH="/usr/local/bin:/usr/local/sbin:$PATH"
brew update
brew install python  # Python 3
sudo pip3 install -U virtualenv
```

- Create a new virtual environment

Ubuntu/WSL/MacOS:
```sh
virtualenv --system-site-packages -p python3 ./venv
```
- Activate the virtual environment:
```sh
source ./venv/bin/activate
```
- Run setup script for installing dependencies and packages.
```sh
script/setup.sh
```
- Make sure you are in the directory `DD2424-project` and download the tiny-imagenet dataset.
```sh
mkdir dataset && cd dataset && wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip && rm tiny-imagenet-200.zip
```
# Run the code on computer with CUDA

- Make sure you have a CUDA 10.1 installed (Tensorflow 2.1 only works on CUDA 10.1). If you are on Ubuntu, this [tutorial](https://www.iridescent.io/tech-blogs-installing-cuda-the-right-way/) is a good one. 

- Install [Docker](https://docs.docker.com/engine/install/ubuntu/)

- Navigate to the repo folder and build the docker image by typing in the following command: 
```sh
docker build --tag tf_alexnet .
```

- When the image is done building run the docker image by typing the following command: 
```sh
docker run --gpus all --rm -it --name tf -v ${PWD}:/workspaces/DD2424-project -p 8080:8080 -p 6060:6060 tf_alexnet bash
```

- You have now opened up the container and can run jupyter by typing the following command:
```sh
jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root
```
- Go to the prompted page to access jupyter. 

- Open the notebook folder and open the notebook "alexnet"

# Run the code on GCP

## On GCP 
- Go to [Google Cloud Platform](https://cloud.google.com/), set up an account and start a new project.
- Create a new deep learning VM and press launch. 

- Set up the VM as you want it. Make sure you select the framework TensorFlow Enterprise 2.1 (CUDA 10.1) and that you check the GPU box.

- Make your external IP static in GCP

- Open the CLI by pressing SSH for your VM and clone this repo: 

```sh
git clone https://github.com/Callet91/DD2424-project.git
```

- Download and unzip tiny-imagenet.

```sh
cd DD2424-project && mkdir dataset && cd dataset && wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip && rm tiny-imagenet-200.zip && cd ..
```

- Make sure [Docker](https://docs.docker.com/engine/install/ubuntu/) is installed.

- Navigate to the repo folder and build the docker image by typing in the following command: 
```sh
docker build --tag tf_alexnet .
```

- When image is done building, run the docker image by typing the following command: 
```sh
docker run --gpus all --rm -it --name tf -v ${PWD}:/workspaces/DD2424-project -p 8888:8888 -p 6060:6060 tf_alexnet
```

- Go to the prompted page to access jupyter. 




