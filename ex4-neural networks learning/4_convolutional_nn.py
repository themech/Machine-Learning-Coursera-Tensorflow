# Here we replace a simple, fully connected neural network with something more suitable for image classification:
# convolutional network. We use two convolutional layers with 5x5 filter. You can read move about this kind of
# networks for example here: http://cs231n.github.io/convolutional-networks/
# It this approach is a little more CPU-expensive that the previous one, we are also introducing batching. In each
# training iteration we will use a subset of our train set (default: 50 images). This helps us iterate quicker.

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

# size of a single digit image (in pixels)
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
TEST_SIZE = 0.25  # test set will be 25% of the data
FSIZE = 5  # Convolutional layer filer size (5x5px)
CONV_LAYERS = 2
CONV1_SIZE = 32
CONV2_SIZE = 64

# Parse the command line arguments (or use default values)
parser = argparse.ArgumentParser(description='Recognizing hand-written number using neural network.')
parser.add_argument('-s', '--fully_connected_layer_size', type=int,
                    help='number of neurons in the densely connected layer (default: 1024)', default=1024)
parser.add_argument('-d', '--dropout', type=float,
                    help='dropout probability (default: 0.5)', default=0.5)
parser.add_argument('-c', '--conv_dropout', type=float,
                    help='convolutional layer dropout (default: 0.95)', default=0.95)
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='learning rate for the algorithm (default: 0.0001)', default=0.0001)
parser.add_argument('-b', '--batch_size', type=int,
                    help='Batch size for a single learning step (default: 50)', default=50)
parser.add_argument('-e', '--epochs', type=int, help='number of epochs (default: 5000)', default=2000)
parser.add_argument('-o', '--optimizer', type=str,
                    help='tensorflow optimizer class (default: AdamOptimizer)', default='AdamOptimizer')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='increase output verbosity')
args = parser.parse_args()

optimizer_class = getattr(tf.train, args.optimizer)

# Load the hand-written digits data.
filename = 'data/ex4data1.mat'
data = io.loadmat(filename)
X_data, Y_data = data['X'], data['y']

# y==10 is digit 0, convert it to 0 then to make the code below simpler
Y_data[Y_data == 10] = 0

# Split the data
X_data, X_test_data, Y_data, Y_test_data = train_test_split(X_data, Y_data, test_size=TEST_SIZE)

# Convert the Y matrixes to 1-D arrays as we're using sparse_softmax_cross_entropy_with_logits
Y_data = Y_data.ravel()
Y_test_data = Y_test_data.ravel()

if args.verbose:
    print 'Shape of the X_data', X_data.shape
    print 'Shape of the Y_data', Y_data.shape
    print 'Shape of the X_test_data', X_test_data.shape
    print 'Shape of the Y_test_data', Y_test_data.shape

numSamples = X_data.shape[0]
numTestSamples = X_test_data.shape[0]


def readout_layer(input, size_in, size_out):
    """Readout layer for our network

    Classifier layer at the end of the neural network. It is similar to the densely connected layer,
    just without the ReLU
    """
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="bias")
    return tf.matmul(input, w) + b


def fc_layer(input, size_in, size_out):
    """Creates a fully connected nn layer.

    The layer is initialized with random numbers from normal distribution. ReLU is applied at the end.
    """
    return tf.nn.relu(readout_layer(input, size_in, size_out))


def conv_layer(input, size_in, size_out):
    """Creates a complete convolutional layer.

    Uses 5x5 px filer, ReLU and max pooling.
    """
    w = tf.Variable(tf.truncated_normal([FSIZE, FSIZE, size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="bias")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Setup placeholders, and reshape the data
x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT])
y = tf.placeholder(tf.int64, shape=[None])  # simple vector

x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

# Experiment - dropout for conv layers. Check if that prevents some overfitting in these layers.
conv_dropout = tf.placeholder(tf.float32)

conv1 = conv_layer(x_image, 1, CONV1_SIZE)  # First convolutional layer
conv1 = tf.nn.dropout(conv1, conv_dropout)
conv2 = conv_layer(conv1, CONV1_SIZE, CONV2_SIZE)  # Second convolutional layer
conv2 = tf.nn.dropout(conv2, conv_dropout)

# Flatten the data before going to the next steps. Conv layer and polling change the dimension, each layer decreases the
# size by half. so the final size can be calculated by rounding up the image_size/(2^conv_layers)
resize_width = int(math.ceil(float(IMAGE_WIDTH) / (2 << (CONV_LAYERS - 1))))
resize_height = int(math.ceil(float(IMAGE_HEIGHT) / (2 << (CONV_LAYERS - 1))))
flattened = tf.reshape(conv2, [-1, CONV2_SIZE * resize_width * resize_height])

# Create a densely connected layer
fc = fc_layer(flattened, CONV2_SIZE * resize_width * resize_height, args.fully_connected_layer_size)

# Apply dropout to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
fc_drop = tf.nn.dropout(fc, keep_prob)

# Read the results, 10 outputs as we have 10 classes (digits)
output_layer = readout_layer(fc_drop, args.fully_connected_layer_size, 10)

# define cost function
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

optimizer = optimizer_class(args.learning_rate).minimize(cost)

# measure accuracy - pick the output with the highest score as the prediction
pred = tf.argmax(output_layer,1)
correct_prediction = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if args.verbose:
    print "Training..."

# Variables for tracking accuracy over time
iter_arr = []
train_accuracy_arr = []
test_accuracy_arr = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def next_batch(num, data, labels):
    """Helper function for creating batches from the training set.

    :param num: number of examples to sample
    :param data: array of features
    :param labels: array of labels
    :return: a tuple: array of sampled features and array of sampled labels
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

for epoch in xrange(args.epochs):
    with sess.as_default():

        if not (epoch + 1) % 15:
            test_accuracy = 0
            train_accuracy = 0
            train_accuracy = accuracy.eval(feed_dict={x: X_data, y: Y_data, keep_prob: 1.0, conv_dropout: 1.0})
            test_accuracy = accuracy.eval(feed_dict={x: X_test_data, y: Y_test_data, keep_prob: 1.0, conv_dropout: 1.0})
            iter_arr.append(epoch)
            train_accuracy_arr.append(train_accuracy)
            test_accuracy_arr.append(test_accuracy)
            if args.verbose:
                print "Epoch:", '%04d' % (epoch + 1), ", accuracy: ", train_accuracy, ", test accuracy: ", test_accuracy

        X_data_batch, Y_data_batch = next_batch(args.batch_size, X_data, Y_data)
        optimizer.run(feed_dict={x: X_data_batch, y: Y_data_batch, keep_prob: args.dropout, conv_dropout: args.conv_dropout})

print "Accuracy report for the learning set"
y_pred = sess.run(pred, feed_dict={x: X_data, y: Y_data, keep_prob: 1.0, conv_dropout: 1.0})
print metrics.classification_report(Y_data, y_pred)

print "Accuracy report for the test set"
y_test_pred = sess.run(pred, feed_dict={x: X_test_data, y: Y_test_data, keep_prob: 1.0, conv_dropout: 1.0})
print "shape of x", X_test_data.shape
print "shape of y", Y_test_data.shape
print "shape of y_test_pred", y_test_pred.shape

print metrics.classification_report(Y_test_data, y_test_pred)

print "Plotting accuracy over time..."
plt.plot(iter_arr, train_accuracy_arr, 'ro-')
plt.plot(iter_arr, test_accuracy_arr, 'g^-')
plt.show()
