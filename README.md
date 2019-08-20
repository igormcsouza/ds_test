# DeeperSystem Test - Computer Vision

Link Task can be found [here](https://gist.github.com/csaftoiu/9fccaf47fd8f96cd378afd8fdd0d63c1).
Link to download the .zip file, which contains the rotated pictures, is also [here](...)

## Little Explanation

The problem is to us CNN to identify images orientation and then, rotate them to the upright position. To solve this problem I use a model given by Keras, the same one to solve the cifar10 problem. The main step of the process was to prepare the data to the model. I used a lib called 'imageio' to read the images and transform them into vectors, after some preprocessing they were ready to be used on the model.

Train the model took actually a lot of time, I adjusted the dropout parameter and then I put 20 epochs and the results were really impressive right on the 3 epoch. The final result was ~97%. Really good. The results is on the folder, right [here](test.preds.csv). I also got the rotated images, they can be found on the link above in a .zip file. As the file was really big, I could not push them togethe with the others.