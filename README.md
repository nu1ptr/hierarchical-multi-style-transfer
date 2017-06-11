# Style Transfer Experimentation

## Features so far
* Single Style Transfer
* Multi Style Transfer
* Transformation Network (In Progress)

## Supported Models
* AlexNet (Not Fully Convolutional)
* VGG16
* VGG19

## Issues
* There is a bug in the ResNet conversion. You cannot run tf.gradients through from your losses to the pixels of the image. See more here: https://github.com/ry/tensorflow-resnet/issues/12

## Dependencies
* opencv 3.20
* numpy
* tensorflow 1.10
* progressbar2
* python 3
