import skimage
from skimage import data
import numpy as np
import tensorflow as tf
import os, sys
import matplotlib.pyplot as plt
from skimage import transform
from skimage.color import rgb2gray
import random
from sklearn.model_selection import train_test_split


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []

    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return np.array(images),np.array(labels)


def conv2d(input, weights, bias, stride= 1):
    x_local= tf.nn.conv2d(input, weights, strides=[1, stride, stride, 1], padding='SAME')
    x_local= tf.nn.bias_add(x_local, bias)
    return tf.nn.relu(x_local)


def maxpool2d(input, k=2):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def convolutional_net(input, weights, biases):

    convolution_1 = conv2d(input, weights['wc1'], biases['bc1'])
    convolution_1 = maxpool2d(convolution_1, k=2)

    convolution_2 = conv2d(convolution_1, weights['wc2'], biases['bc2'])
    convolution_2 = maxpool2d(convolution_2, k=2)

    convolution_3 = conv2d(convolution_2, weights['wc3'], biases['bc3'])
    convolution_3 = maxpool2d(convolution_3, k=2)

    fully_con = tf.reshape(convolution_3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fully_con = tf.add(tf.matmul(fully_con, weights['wd1']), biases['bd1'])
    fully_con = tf.nn.relu(fully_con)

    output = tf.add(tf.matmul(fully_con, weights['out']), biases['out'])

    return output


ROOT_PATH = "D:\python projects\ProiectImagine"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns\Testing")

images,labels = load_data(train_data_directory)
unique_labels= set(labels)

images28 = [transform.resize(image,(28,28)) for image in images]
images28 = np.array(images28)
#images28 = np.array(rgb2gray(images28))


images28=images28.reshape(-1,28,28,1)
b=np.zeros((len(labels),62))
lungimea=len(labels)
b[np.arange(lungimea),labels] = 1
labels = b

print("Forma: ",labels.shape)
training_iters = 200
learning_rate = 0.001
batch_size = 128


x = tf.placeholder(dtype=tf.float32, shape = [None, 28, 28,1])
y = tf.placeholder(dtype=tf.int32, shape = [None, 62])

weights = {
    'wc1': tf.get_variable('W0', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(4 * 4 * 128, 128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128, 62), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(62), initializer=tf.contrib.layers.xavier_initializer()),
}

prediction = convolutional_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


sesiunea = tf.Session()

sesiunea.run(tf.global_variables_initializer())



for i in range(201):
    lossu = 0.0
    acuratu = 0.0
    #print("EPOCH: ",i)
    for batch in range(len(images28) // batch_size):
        batch_x = images28[batch * batch_size:min((batch + 1) * batch_size, len(images28))]
        batch_y = labels[batch * batch_size:min((batch + 1) * batch_size, len(labels))]
        opt = sesiunea.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        loss_val, accuracy_val = sesiunea.run([cost,accuracy] , feed_dict={x: batch_x, y: batch_y})
        lossu=loss_val
        acuratu=accuracy_val
    if i%1 == 0:
        print("Epoch: ",i,"Loss: ", loss_val, " Acurracy: ", accuracy_val)

sesiunea.close()
