# Style Transfer Experimentation
![Style Transfer](/images/1.png)

## Features so far
* Single Style Transfer
* Multi Style Transfer (Might have to find really good hyper-parameters to get this to work. There is probably a "pull and tug" on the loss between the different styles, so won't converge well)
* Transformation Network (In Progress)

## Supported Models
* AlexNet ,npy (Not Fully Convolutional)
* VGG16	,tfmodel
* VGG19 ,npy

## Issues
* There is a bug in the ResNet conversion. You cannot run tf.gradients from your losses to the pixels of the image with resnet. See more here: https://github.com/ry/tensorflow-resnet/issues/12
	* Try converting with Kaffe instead of Ry's converter
* Occasional saturation on images

## Dependencies
* opencv 3.20
* numpy
* tensorflow 1.10
* progressbar2
* python 3
