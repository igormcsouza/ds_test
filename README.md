# DeeperSystem Test - Computer Vision

Link Task can be found [here](https://gist.github.com/csaftoiu/9fccaf47fd8f96cd378afd8fdd0d63c1).
Link to download the .zip file, which contains the rotated pictures, is also [here](...)

## Little Explanation

The problem is to us CNN to identify images orientation and then, rotate them to the upright position. To solve this problem I use a model given by Keras, the same one to solve the cifar10 problem. The main step of the process was to prepare the data to the model. I used a lib called 'imageio' to read the images and transforme them into vectors, after some preprocessing they were ready to be used on the model.