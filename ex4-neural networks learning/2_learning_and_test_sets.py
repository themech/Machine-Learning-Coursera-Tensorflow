# This exercise is very similar to the previous one. The only difference is
# that we will get better overview of how the neural network is learning. To do
# this we will split out input data into learning set and test set. We will
# perform all the learning using the first set of data. Of course as the number
# of epochs increases, so does the accuracy on the learning set (as the cost
# function on this set is minimized by the learning process). But every now and
# then we will check the accuracy on the test set - data that the network
# hasn't seen during the learning phase. Ideally the accuracy on the learning
# set should increase along with the accuracy on the test set, as this means
# the network has "learned" something general that can be applied on a data
# that was not seen. At the end we will draw a little chart to check how the
# accuracy was changing over time for both sets.
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
parser = argparse.ArgumentParser(
    description='Recognizing hand-written number using neural network.')
parser.add_argument('-s', '--hidden_layer_size', type=int,
                    help='number of neurons in the hidden layer (default: 64)',
                    default=64)
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='learning rate for the algorithm (default: 0.5)',
                    default=0.5)
parser.add_argument('-d', '--decay', dest='decay', type=float,
                    help='learning rate decay (default: 0.9999, 1.0 means '
                    'no decay)', default=0.9999)
parser.add_argument('-e', '--epochs', type=int,
                    help='number of epochs (default: 1000)', default=1000)
parser.add_argument('-o', '--optimizer', type=str,
                    help='tensorflow optimizer class (default: '
                    'AdagradOptimizer)', default='AdagradOptimizer')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='increase output verbosity')
args = parser.parse_args()

optimizer_class = getattr(tf.train, args.optimizer)

# Load the hand-written digits data.
filename = 'data/ex4data1.mat'
data = io.loadmat(filename)
X_data, Y_data = data['X'], data['y']

# y==10 is digit 0, convert it to 0 then to make the code below simpler
Y_data[Y_data == 10] = 0

# Split the data
X_data, X_test_data, Y_data, Y_test_data = train_test_split(
    X_data, Y_data, test_size=TEST_SIZE)

if args.verbose:
    print('Shape of the X_data', X_data.shape)
    print('Shape of the Y_data', Y_data.shape)
    print('Shape of the X_test_data', X_test_data.shape)
    print('Shape of the Y_test_data', Y_test_data.shape)

numSamples = X_data.shape[0]
numTestSamples = X_test_data.shape[0]


def fc_layer(input, size_in, size_out):
    """Creates a fully connected nn layer.

    The layer is initialized with random numbers from normal distribution.
    """
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1))
    b = tf.Variable(tf.truncated_normal([size_out], stddev=0.1))
    return tf.nn.relu(tf.matmul(input, w) + b)


# Setup placeholders, and reshape the data
x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT])
# 10 outputs, one for each digit
y = tf.placeholder(tf.float32, shape=[None, 10])

if args.verbose:
    print("Creating a network with {:d} neurons in a hidden layer".format(
        args.hidden_layer_size))

hidden_layer = fc_layer(x, IMAGE_WIDTH * IMAGE_HEIGHT, args.hidden_layer_size)
output_layer = fc_layer(hidden_layer, args.hidden_layer_size, 10)

# define cost function and
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=output_layer, labels=y))
# learning rate decay
batch = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
  args.learning_rate,  # Base learning rate.
  batch,               # Current index into the dataset.
  1,                   # Decay step.
  args.decay,          # Decay rate.
  staircase=True)

optimizer = optimizer_class(learning_rate).minimize(cost, global_step=batch)

# measure accuracy - pick the output with the highest score as the prediction
pred = tf.argmax(tf.nn.softmax(output_layer), 1)  # softmax is optional here
correct_prediction = tf.equal(pred, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Convert the aswers vector to a sparse matrix (refer to
# 1_nn_training_example.py for a more detailed comment)
Y_sparse = tf.keras.utils.to_categorical(Y_data, 10)
Y_test_sparse = tf.keras.utils.to_categorical(Y_test_data, 10)

print("Training...")

# Variables for tracking accuracy over time
iter_arr = []
train_accuracy_arr = []
test_accuracy_arr = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(args.epochs):
    if not (epoch+1) % 5:
        train_accuracy = sess.run([accuracy],
                                  feed_dict={x: X_data, y: Y_sparse})
        test_accuracy = sess.run([accuracy],
                                 feed_dict={x: X_test_data, y: Y_test_sparse})
        iter_arr.append(epoch)
        train_accuracy_arr.append(train_accuracy)
        test_accuracy_arr.append(test_accuracy)
        if args.verbose:
            print('Epoch: {:04d}, accuracy: {}, test accuracy: {}'.format(
                epoch+1, train_accuracy, test_accuracy))
    sess.run([optimizer], feed_dict={x: X_data, y: Y_sparse})

print("Accuracy report for the learning set")
y_pred = sess.run(pred, feed_dict={x: X_data, y: Y_sparse})
print(metrics.classification_report(Y_data, y_pred))

print("Accuracy report for the test set")
y_test_pred = sess.run(pred, feed_dict={x: X_test_data, y: Y_test_sparse})
print(metrics.classification_report(Y_test_data, y_test_pred))

print("Plotting accuracy over time...")
plt.plot(iter_arr, train_accuracy_arr, label='train accuracy')
plt.plot(iter_arr, test_accuracy_arr, label='test accuracy')
plt.xlabel('epoch', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)

plt.show()
