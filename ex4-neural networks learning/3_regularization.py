# In this example we're adding simple regularization to try prevent overfitting.
# Regularization is implemented by minimizing non-bias NN variables.
# By playing with the "r" parameter you can see that it can decrease the difference between lean and test set accuracy.
# Unfortunately it doesn't produce better much better results, as fully connected network is not best suited
# for recognizing images. Also regularization that works for linear regression is not best suited for deep networks as
# they are highly nonconvex.
# In te next example we will implement convolutional network that is able to look for specific shapes in the image
# rather than pixels.

import argparse
import matplotlib.pyplot as plt
from scipy import io
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf

# size of a single digit image (in pixels)
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
TEST_SIZE = 0.25  # test set will be 25% of the data

# Parse the command line arguments (or use default values).
parser = argparse.ArgumentParser(description='Recognizing hand-written number using neural network.')
parser.add_argument('-s', '--hidden_layer_size', type=int,
                    help='number of neurons in the hidden layer (default: 25)', default=25)
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='learning rate for the algorithm (default: 0.0001)', default=0.0001)
parser.add_argument('-r', '--regularizer', type=float,
                    help='regularization multiplier (default: 0.001)', default=0.001)
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

# Split the data
X_data, X_test_data, Y_data, Y_test_data = train_test_split(X_data, Y_data, test_size=TEST_SIZE)

# In this example, instead of playing with sparse Y vectors, we simply keep the class labels and
# use sparse_softmax_cross_entropy_with_logits later as a cost function.
Y_data = Y_data.ravel()
Y_test_data = Y_test_data.ravel()

if args.verbose:
    print 'Shape of the X_data', X_data.shape
    print 'Shape of the Y_data', Y_data.shape
    print 'Shape of the X_test_data', X_test_data.shape
    print 'Shape of the Y_test_data', Y_test_data.shape

numSamples = X_data.shape[0]
numTestSamples = X_test_data.shape[0]


def fc_layer(input, size_in, size_out):
    """Creates a fully connected nn layer.

    The layer is initialized with random numbers from normal distribution.
    """
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name='bias')  # name needed later for filtering
    return tf.nn.relu(tf.matmul(input, w) + b)

# Setup placeholders, and reshape the data
x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT])
y = tf.placeholder(tf.int64, shape=[None])  # simple vector, output is the class number

if args.verbose:
    print "Creating a network with %d neurons in a hidden layer" % args.hidden_layer_size

hidden_layer = fc_layer(x, IMAGE_WIDTH * IMAGE_HEIGHT, args.hidden_layer_size)
output_layer = fc_layer(hidden_layer, args.hidden_layer_size, 10)

# calculate the regularization cost by combining non-bias variables
nn_vars = tf.trainable_variables()
regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in nn_vars if 'bias' not in v.name]) * args.regularizer
# define cost function
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

optimizer = optimizer_class(args.learning_rate).minimize(cost + regularization_loss)

# measure accuracy - pick the output with the highest score as the prediction
pred = tf.argmax(output_layer, 1)
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

for epoch in xrange(args.epochs):

    if not epoch % 5:
        train_accuracy = sess.run([accuracy], feed_dict={x: X_data, y: Y_data})
        test_accuracy = sess.run([accuracy], feed_dict={x: X_test_data, y: Y_test_data})
        iter_arr.append(epoch)
        train_accuracy_arr.append(train_accuracy)
        test_accuracy_arr.append(test_accuracy)
        if args.verbose:
            print "Epoch:", '%04d' % (epoch + 1), ", accuracy: ", train_accuracy, ", test accuracy: ", test_accuracy
    sess.run([optimizer], feed_dict={x: X_data, y: Y_data})

print "Accuracy report for the learning set"
y_pred = sess.run(pred, feed_dict={x: X_data, y: Y_data})
print metrics.classification_report(Y_data, y_pred)

print "Accuracy report for the test set"
y_test_pred = sess.run(pred, feed_dict={x: X_test_data, y: Y_test_data})
print metrics.classification_report(Y_test_data, y_test_pred)

print "Plotting accuracy over time..."
plt.plot(iter_arr, train_accuracy_arr, 'ro-')
plt.plot(iter_arr, test_accuracy_arr, 'g^-')
plt.show()
