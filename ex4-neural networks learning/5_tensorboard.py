# So far we've been making our own plots to visualize accuracy over time.
# Tensorflow comes with a tool tensorboard in which we can visualize merge_all
# the variables (including weights and biases) and even data points.
# In this exercie we will use TF summary writers to dump variables while the
# model is learaning. Also for the test set we will generate metadata (labels
# and images) so we can display them on 3d embeddings in tensorboard.
# Usage:
# train the model: python 5_tensorboard.py --verbose
# run tensorboard: tensorboard --logdir=/tmp/logdir/
# inspect the data: open http://localhost:6006 in your browser
import argparse
import math
import numpy as np
import os
from scipy import io, misc, sparse
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
import scipy

# size of a single digit image (in pixels)
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
TEST_SIZE = 0.25  # test set will be 25% of the data

# Parse the command line arguments (or use default values).
parser = argparse.ArgumentParser(
    description='Recognizing hand-written number using neural network.')
parser.add_argument('-s', '--hidden_layer_size', type=int,
                    help='number of neurons in the hidden layer (default: 25)',
                    default=25)
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='learning rate for the algorithm (default: 0.0001)',
                    default=0.0001)
parser.add_argument('-e', '--epochs', type=int,
                    help='number of epochs (default: 5000)', default=5000)
parser.add_argument('-o', '--optimizer', type=str,
                    help='tensorflow optimizer class (default: AdamOptimizer)',
                    default='AdamOptimizer')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='increase output verbosity')
parser.add_argument('--dir', type=str,
                    help='directory to store the training process summary in '
                    '(default: /tmp/logdir)', default='/tmp/logdir')
args = parser.parse_args()

optimizer_class = getattr(tf.train, args.optimizer)

# Load the hand-written digits data.
filename = 'data/ex4data1.mat'
data = io.loadmat(filename)
X_data, Y_data = data['X'], data['y']

# y==10 is digit 0, convert it to 0 then to make the code below simpler
Y_data[Y_data == 10] = 0

X_data, X_test_data, Y_data, Y_test_data = train_test_split(
    X_data, Y_data, test_size=TEST_SIZE)

numSamples = X_data.shape[0]
numTestSamples = X_test_data.shape[0]


def fc_layer(input, size_in, size_out, name="fc"):
    """Creates a fully connected nn layer.

    The layer is initialized with random numbers from normal distribution.
    """
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1),
                        name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


# Setup placeholders, and reshape the data
# While building the model, name all the blocks and variables so they're
# easier to identify in tensorboard
x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH * IMAGE_HEIGHT],
                   name="x")
y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

if args.verbose:
    print("Creating a network with {:d} neurons in a hidden layer".format(
        args.hidden_layer_size))

hidden_layer = fc_layer(x, IMAGE_WIDTH * IMAGE_HEIGHT, args.hidden_layer_size,
                        "hidden_layer")
output_layer = fc_layer(hidden_layer, args.hidden_layer_size, 10,
                        "output_layer")

with tf.name_scope("xent"):
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=output_layer, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(xent)

with tf.name_scope("accuracy"):
    pred = tf.argmax(output_layer, 1)
    correct_prediction = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Prepare the summary writer dir
if not os.path.isdir(args.dir):
    os.mkdir(args.dir)

# In this section we will prepare tensorboard embeddings for the test set.
# Embedding metadata: generate a TSV file with labels
labels_filename = os.path.join(args.dir, 'labels.tsv')
with open(labels_filename, 'w') as labels_file:
    for i in range(numTestSamples):
        labels_file.write("%d\n" % Y_test_data[i])


def get_sprites(X):
    """Generate a square sprite image with digits.
    Tensorflow supports sprite images up to 8192x8192 pixels. Given each of
    our digits is 20px*20px, we don't have to scale it down.
    """
    width, height = IMAGE_WIDTH, IMAGE_HEIGHT
    digits_per_row = int(math.ceil(math.sqrt(X.shape[0])))
    nrows, ncols = digits_per_row, digits_per_row

    big_picture = np.zeros((height * nrows, width * ncols))

    irow, icol = 0, 0
    for idx in range(X.shape[0]):
        if icol == ncols:
            irow += 1
            icol = 0
        iimg = X[idx].reshape(width, height).T  # transpose the data set
        big_picture[irow * height:irow * height + iimg.shape[0],
                    icol * width:icol * width + iimg.shape[1]] = iimg
        icol += 1
    return misc.toimage(big_picture)


# Generate images associated with the embeddings.
sprite_img = get_sprites(X_test_data)
sprite_filename = os.path.join(args.dir, 'sprite.png')
scipy.misc.imsave(sprite_filename, sprite_img)

# Create embeddings and assigment operation to capture the test set data
fc_embedding = tf.Variable(tf.zeros([numTestSamples, args.hidden_layer_size]),
                           name="test_embedding_hidden_layer")
fc_assignment = fc_embedding.assign(hidden_layer)
logits_embedding = tf.Variable(tf.zeros([numTestSamples, 10]),
                               name="test_embedding_logits")
logits_assignment = logits_embedding.assign(output_layer)

# Create two embeddings - for hidden layer and for logits
projector_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding = projector_config.embeddings.add()
embedding.tensor_name = fc_embedding.name
embedding.metadata_path = labels_filename
embedding.sprite.image_path = sprite_filename
embedding.sprite.single_image_dim.extend([IMAGE_WIDTH, IMAGE_HEIGHT])

embedding = projector_config.embeddings.add()
embedding.tensor_name = logits_embedding.name
embedding.metadata_path = labels_filename
embedding.sprite.image_path = sprite_filename
embedding.sprite.single_image_dim.extend([IMAGE_WIDTH, IMAGE_HEIGHT])

# Summary operation for writing down variables values
summary = tf.summary.merge_all()

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter(os.path.join(args.dir, 'train'),
                                     sess.graph)
test_writer = tf.summary.FileWriter(os.path.join(args.dir, 'test'), sess.graph)

tf.contrib.tensorboard.plugins.projector.visualize_embeddings(test_writer,
                                                              projector_config)

# Convert the aswers vector to a sparse matrix (refer to
# 1_nn_training_example.py for a more detailed comment)
Y_sparse = sparse.csr_matrix((np.ones(numSamples),
                              Y_data.reshape(numSamples),
                              range(numSamples+1))).toarray()
Y_test_sparse = sparse.csr_matrix((np.ones(numTestSamples),
                                   Y_test_data.reshape(numTestSamples),
                                   range(numTestSamples+1))).toarray()

print("Training...")

for epoch in range(args.epochs):

    if not epoch % 5:
        # Write summary for the training set
        train_accuracy, s = sess.run([accuracy, summary],
                                     feed_dict={x: X_data, y: Y_sparse})
        train_writer.add_summary(s, epoch)
        if not epoch % 20:
            # Write summary for the test set
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            test_accuracy, s, y_test_pred = sess.run(
                [accuracy, summary, pred],
                feed_dict={x: X_test_data, y: Y_test_sparse},
                options=run_options,
                run_metadata=run_metadata)
            test_writer.add_run_metadata(run_metadata,
                                         'step{:03d}'.format(epoch))
            test_writer.add_summary(s, epoch)

            if args.verbose:
                print('Epoch: {:04d}, accuracy: {}, test accuracy: {}'.format(
                    epoch+1, train_accuracy, test_accuracy))

    if not epoch % 100:
        # Projector data in tensorboard is based on checkpoints, so every now
        # and then capture the embeddings data and save the model
        sess.run([fc_assignment, logits_assignment],
                 feed_dict={x: X_test_data, y: Y_test_sparse})

        saver.save(sess, os.path.join(args.dir, 'model.ckpt'), epoch)
    _, y_pred = sess.run([train_step, pred],
                         feed_dict={x: X_data, y: Y_sparse})

train_writer.close()
test_writer.close()

print(metrics.classification_report(Y_data, y_pred))
print(metrics.classification_report(Y_test_data, y_test_pred))
