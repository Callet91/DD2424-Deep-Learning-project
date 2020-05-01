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

The group is aiming for the grade D.

## Initial experiments
The initial goal is to successfully set up AlexNet in classifying the images in the ImageNet dataset. The two initial experiments will be to create an average object for each label of the trained network and to measure the accuracy of the system.

## Measurement of success
In order to ensure that the code works as intended, a Test Driven Development (TDD) approach will be used. This enables the project members to work in parallel with the development of the code. For the integration part, a simple Continuous Integration (CI) pipeline will be used in Jenkins. This environment will enable both parallel development and ensure that each separate part of the code works as intended and that integration of every part works as well.

To measure the performance of the network, the network will be trained, validated and tested on separate datasets from a subset of the ImageNet. The results will be compared with the results achieved by students from the course cs231 held at Stanford.

## Getting started (easy way)

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

## References
Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, 2012,
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

 ILSVRC: imagenet large scale visual recognition competition. http://www.image-net.org/ challenges/LSVRC/
