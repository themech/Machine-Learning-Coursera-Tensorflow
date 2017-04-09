import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import io, misc
import tensorflow as tf

# Parse the command line arguments (or use default values).
parser = argparse.ArgumentParser(description='Recognizing hand-written number using multiclass logistic regression.')
parser.add_argument('-lr', '--learning_rate',type=float,
                    help='learning rate for the algorithm (default: 0.1)', default=0.1)
parser.add_argument('-r', '--regularization',type=float,
                    help='theta regularization value (default: 0.1)', default=0.1)
parser.add_argument('-e', '--epochs', type=int, help='number of epochs (default: 400)', default=400)
parser.add_argument('-o', '--optimizer', type=str,
                    help='tensorflow optimizer class (default: AdamOptimizer)', default='AdamOptimizer')
# other optimizers to try out: GradientDescentOptimizer, AdadeltaOptimizer, AdagradOptimizer, AdamOptimizer,
# FtrlOptimizer, RMSPropOptimizer

parser.add_argument('--verbose', dest='verbose', action='store_true', help='increase output verbosity')
parser.add_argument('--silent', dest='verbose', action='store_false')
parser.set_defaults(verbose=True)
args = parser.parse_args()

optimizer_class = getattr(tf.train, args.optimizer)

# Load the hand-written digits data.
filename = 'data/ex3data1.mat'
data = io.loadmat(filename)
X_data, Y_data = data['X'], data['y']

# y==10 is digit 0, convert it to 0 then to make the code below simpler
Y_data[Y_data==10] = 0

numSamples = X_data.shape[0]

if args.verbose:
    print 'Shape of the X_data', X_data.shape
    print 'Shape of the Y data', Y_data.shape

def plot_100_images(X):
    """Plot 100 randomly picked digits."""
    width, height = 20, 20
    nrows, ncols = 10, 10
    indices_to_display = np.random.choice(range(X.shape[0]), nrows * ncols)

    big_picture = np.zeros((height * nrows, width * ncols))

    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = X[idx].reshape(width, height).T # transpose the data set
        big_picture[irow * height:irow * height + iimg.shape[0], icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    fig = plt.figure(figsize=(6, 6))
    img = misc.toimage(big_picture)
    plt.imshow(img, cmap=matplotlib.cm.Greys_r)

    plt.show()

if args.verbose:
    # Plot some of the loaded digits.
    plot_100_images(X_data)

# For each row, add a constant (1) at the beginning, needed for logistic regression.
X_data = np.insert(X_data, 0, 1, axis=1)

def logistic_regression(X_data, Y_data, optimizer_class, reg, learning_rate, epochs, verbose=True):
    """
    Trains and returns a classifier that recognizes one digit (although the code below is fairly general).
    :param X_data: Our digit data (learning set)
    :param Y_data: Digit label, 1 for row containing the digit we're trying to learn to recognize, 0 for others
    :param optimizer_class: class that will be used to create optimizer object.
    :param reg: regularization parameter
    :param learning_rate: learning rate parameter
    :param epochs: number of epochs
    :param verbose: whether to print out some debugging information
    :return: Trained classifier that can be used to classify digits.
    """
    numFeatures = X_data.shape[1]
    numSamples = X_data.shape[0]
    X = tf.placeholder(tf.float32, shape=[None, numFeatures])
    Y = tf.placeholder(tf.float32, shape=[None, 1])
    W = tf.Variable(tf.zeros([numFeatures, 1]))
    pred = tf.nn.sigmoid(tf.matmul(X, W))
    cost = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(pred, 1e-9, 1)) +
                          (1 - Y) * tf.log(tf.clip_by_value(1 - pred, 1e-9, 1))) / numSamples
    regularized_W = tf.slice(W, [1, 0], [-1, -1])  # don't regularize W[0]
    regularizer = tf.reduce_sum(tf.square(regularized_W)) * reg / numFeatures
    correct_predict = tf.equal(tf.cast(tf.greater(pred, 0.5), tf.float32), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

    optimizer = optimizer_class(learning_rate).minimize(cost + regularizer)
    init = tf.global_variables_initializer()

    # Create a tensorflow session
    sess = tf.Session()
    sess.run(init)
    for epoch in range(epochs):
        _, cost_value, reg_cost, accuracy_value = sess.run([optimizer, cost, regularizer, accuracy],
                                                                    feed_dict={X: X_data, Y: Y_data})
        # Display logs per epoch step
        if verbose and (epoch + 1) % 50 == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value), \
                "reg=", "{:.9f}".format(reg_cost), "accuracy=", accuracy_value

    classifier = tf.greater(pred, 0.5)
    return lambda X_data: sess.run([pred, classifier], feed_dict={X: X_data})

classifiers = []  # This will hold our 10 classifiers, one for each digit.
for k in range(10):
    Yk_data = (Y_data == k).astype(int)  # prepare the labels for the current digit.
    tk = logistic_regression(X_data, Yk_data, optimizer_class, args.regularization,
                             args.learning_rate, args.epochs, args.verbose)
    classifiers.append(tk)

# Now we're using each of the classifiers to estimate how much a given row reassembles the digit it tried to learn.
predictions = []
for t in classifiers:
    # Classifier returns 2 values, a score and a boolean (if the score is above 0.5). We need just the score
    # as we treat it as a confidence level.
    pred, _ = t(X_data)
    predictions.append(pred)

# For each row, merge the predictions from all the classifiers. Pick the classifier with the highest confidence level.
prob_matrix = np.concatenate(predictions, axis=1)
y_pred = np.argmax(prob_matrix, axis=1)

if args.verbose:
    print y_pred
    print Y_data

# Print the final report
from sklearn import metrics
print "Optimizer %s, epochs %d, learning_rate %0.2f, regularization param %0.2f" % (args.optimizer, args.epochs,
                                                                                    args.learning_rate,
                                                                                    args.regularization)
print metrics.classification_report(Y_data, y_pred)
