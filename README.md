# Traffic-signs-recognition-with-Convolutional-Neural-Network-in-TensorFlow

This is an implementation of a Convolutional Neural Network with TensorFlow. The network should learn how to clasify traffic signs into 62 classes. I am still working on it, in this moment it doesn't learn very good, not a high succes rate. I want to try another arhitecture like AlexNet or VG16, increase input images dimensions, and maybe make it learn from rgb images, in this stage, for learning are used only gray images.

As optimizer I used Adam and for loss function Binary cross emtropy with logits.

First try:
I used images of size 98x98 in rgb format. The architecture is a simple one:
1. Convolutional Layer with 32 filters of dimensions 3x3 and stride 2
2. Max Pooling Layer of dimensions 2x2 and stride 2
3. Convolutional Layer with 64 filters of dimensions 3x3 and stride 2
4. Max Pooling Layer of dimensions 2x2 and stride 2
5. Convolutional Layer with 128 filters of dimensions 3x3 and stride 2
6. Max Pooling Layer of dimensions 2x2 and stride 2
7. Fully Connected Layer with 128 Perceptrons and input of dimension 5 * 5 * 128
8. Fully Connected Layer with 62 Perceptrons

Performances obatined:
