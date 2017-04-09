# Training a very simple neural network for hand-written digits recognition.
#
# This loads the images of hand-written digits from a data/ex4data1.mat file. Each digits is a described by 401 numbers.
# 400 numbers represent the digit image (20x20px) and the last number is the digit label (the correct answer).
#
# We will be using a network with one hidden layer (with 25 neurons - this can be changed using a command line param).
# So the network has 400 inputs (one for each pixel), a hidden layer and 10 outputs (one for each digit).

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import io, misc, sparse
from sklearn import metrics
import tensorflow as tf

# size of the a single digit image (in pixels)
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20

# Parse the command line arguments (or use default values).
parser = argparse.ArgumentParser(description='Recognizing hand-written number using neural network.')
parser.add_argument('-s', '--hidden_layer_size', type=int,
                    help='number of neurons in the hidden layer (default: 25)', default=25)
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='learning rate for the algorithm (default: 0.0001)', default=0.0001)
parser.add_argument('-e', '--epochs', type=int, help='number of epochs (default: 5000)', default=5000)
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

if args.verbose:
    print 'Shape of the X_data', X_data.shape
    print 'Shape of the Y data', Y_data.shape


def plot_100_images(X, indices=None):
    """Plot 100 randomly picked digits."""
    width, height = IMAGE_WIDTH, IMAGE_HEIGHT
    nrows, ncols = 10, 10
    if indices is None:
        indices = range(X.shape[0])
    indices_to_display = np.random.choice(indices, nrows * ncols)

    big_picture = np.zeros((height * nrows, width * ncols))

    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = X[idx].reshape(width, height).T  # transpose the data set
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    img = misc.toimage(big_picture)
    plt.imshow(img, cmap=matplotlib.cm.Greys_r)

    plt.show()

if args.verbose:
    # Plot some of the loaded digits.
    print "Drawing 100 random digits from the input data"
    plot_100_images(X_data)

numSamples = X_data.shape[0]


def fc_layer(input, size_in, size_out):
    """Creates a fully connected nn layer.

    The layer is initialized with random numbers from normal distribution.
    """
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]))
    return tf.nn.relu(tf.matmul(input, w) + b)

# Setup placeholders, and reshape the data
x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT])
y = tf.placeholder(tf.float32, shape=[None, 10])  # 10 outputs, one for each digit

if args.verbose:
    print "Creating a network with %d neurons in a hidden layer" % args.hidden_layer_size

hidden_layer = fc_layer(x, IMAGE_WIDTH * IMAGE_HEIGHT, args.hidden_layer_size)
output_layer = fc_layer(hidden_layer, args.hidden_layer_size, 10)

# define cost function and
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = optimizer_class(args.learning_rate).minimize(cost)

# measure accuracy - pick the output with the highest score as the prediction
pred = tf.argmax(output_layer, 1)
correct_prediction = tf.equal(pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Y_data is a 1-column vector with the correct answer (digit) in each row. As our neural network has 10 outputs (one
# for each digit) we have to convert Y_data to a sparse matrix. So each row in converted from a single digit to a
# 10-digit vector having nine zeros and a single number one (indicating the correct answer for a given row/image)
Y_sparse = sparse.csr_matrix((np.ones(numSamples), Y_data.reshape(numSamples), range(numSamples+1))).toarray()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if args.verbose:
    print "Training..."

for epoch in xrange(args.epochs):
    _, accuracy_value, cost_value = sess.run([optimizer, accuracy, cost], feed_dict={x: X_data, y: Y_sparse})
    if args.verbose and not epoch % 20:
        print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value), \
            "accuracy=", accuracy_value

# Get the answers (from the nn) and print the accuracy report
y_pred = sess.run(pred, feed_dict={x: X_data, y: Y_sparse})
print metrics.classification_report(Y_data, y_pred)

if args.verbose:
    print "Drawing 100 digits classified as '5'"
    indices = []
    for i in range(len(y_pred)):
        if y_pred[i] == 5:
            indices.append(i)
    plot_100_images(X_data, indices)
