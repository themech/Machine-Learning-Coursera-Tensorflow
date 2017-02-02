import matplotlib.pyplot as plt
import numpy
import pandas as pd
import tensorflow as tf

# Parameters
display_step = 50
learning_rate = 0.01
training_epochs = 1000

raw_data = pd.read_csv('data/ex1data2.txt', names=['square', 'bedrooms', 'price'])

X_data = raw_data[['square', 'bedrooms']]
Y_data = raw_data[['price']]

# Normalize the features
X_data_mean = numpy.mean(X_data)
X_data_std = numpy.std(X_data)
X_data = (X_data - X_data_mean) / X_data_std

n_samples, n_features = X_data.shape

# tf Graph Input
X = tf.placeholder('float', shape=X_data.shape)
Y = tf.placeholder('float', shape=Y_data.shape)

# Set model weights
W = tf.Variable(tf.zeros([n_features, 1]), name='weight')
b = tf.Variable(tf.zeros(1), name='bias')

# Construct a linear model
pred = tf.add(tf.matmul(X, W), b)

# Mean squared error
cost = tf.reduce_mean(tf.square(pred-Y)) / 2.0

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    cost_value, w_value, b_value = (0.0, 0.0, 0.0)
    steps = numpy.array([])
    J_hist = numpy.array([])
    for epoch in range(training_epochs):
        # Fit all training data
        _, cost_value, w_value, b_value = sess.run((optimizer, cost, W, b), feed_dict={X: X_data, Y: Y_data})

        steps = numpy.append(steps, [epoch])
        J_hist = numpy.append(J_hist, [cost_value])

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print 'Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(cost_value), \
                'W=', w_value, 'b=', b_value

    print 'Optimization Finished!'
    print 'Training cost=', cost_value, 'W=', w_value, 'b=', b_value, '\n'

    # Graphic display
    plt.figure(figsize=(6, 5))
    plt.plot(steps, J_hist)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Estimate the price - remember about normalization!
    priceEst = [1, 1650, 3]
    temp = [(i - j) for (i, j) in zip(priceEst, [0, X_data_mean[0], X_data_mean[1]])]
    price = numpy.array([(a / b) for (a, b) in zip(temp, [1, X_data_std[0], X_data_std[1]])])

    theta = numpy.reshape([b_value[0], w_value[0], w_value[1]], (3, 1))
    print price
    print theta
    print 'Predicted price of a 1650 sq-ft, 3 br house (uisng normal equations): $%.2f\n' % (numpy.dot(price, theta)[0])

# Learning rate
base = numpy.logspace(-1, -5, num=4)
lr_candidates = numpy.sort(numpy.concatenate((base, base*3)))
print lr_candidates
training_epochs = 80

fig, ax = plt.subplots(figsize=(16, 9))
for learning_rate in lr_candidates:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(init)
        cost_data = numpy.array([])
        for epoch in range(training_epochs):
            # Fit all training data
            _, cost_value = sess.run((optimizer, cost), feed_dict={X: X_data, Y: Y_data})
            cost_data = numpy.append(cost_data, cost_value)
        ax.plot(numpy.arange(training_epochs), cost_data, label=learning_rate)

ax.set_xlabel('epoch', fontsize=18)
ax.set_ylabel('cost', fontsize=18)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_title('learning rate', fontsize=18)
plt.show()
