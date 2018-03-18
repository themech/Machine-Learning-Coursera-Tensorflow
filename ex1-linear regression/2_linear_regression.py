import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Parameters
display_step = 50
learning_rate = 0.01
training_epochs = 1000

data = pd.read_csv('data/ex1data1.txt', names=['population', 'profit'])

X_data = data[['population']]
Y_data = data[['profit']]

n_samples = X_data.shape[0]  # Number of rows

# tf Graph Input
X = tf.placeholder('float', shape=X_data.shape)
Y = tf.placeholder('float', shape=Y_data.shape)

# Set model weights
W = tf.Variable(tf.zeros([1, 1]), name='weight')
b = tf.Variable(tf.zeros(1), name='bias')

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
# cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
cost = tf.reduce_mean(tf.square(pred-Y)) / 2.0

# Gradient descent
# may try other optimizers like AdadeltaOptimizer, AdagradOptimizer,
# AdamOptimizer, FtrlOptimizer or RMSPropOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    cost_value, w_value, b_value = (0.0, 0.0, 0.0)
    for epoch in range(training_epochs):
        # Fit all training data
        _, cost_value, w_value, b_value = sess.run(
            (optimizer, cost, W, b),
            feed_dict={X: X_data, Y: Y_data})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print('Epoch: {:04d} cost={:.9f} W={} b={}'.format(
                epoch+1, cost_value, w_value, b_value))

    print('Optimization Finished!')
    print('Training cost={:.9f} W={} b={}'.format(
        cost_value, w_value, b_value))

    # Graphic display
    plt.plot(X_data, Y_data, 'ro', label='Original data')
    plt.plot(X_data, w_value * X_data + b_value, label='Fitted line')
    plt.legend()
    plt.show()
