import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# V1. Print data sample
df = pd.read_csv('data/ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
print(df.shape)
print df.head()

# 2. Visualize the data
sns.set(context='notebook', style='darkgrid', palette=sns.color_palette('RdBu', 2))
sns.lmplot('exam1', 'exam2', hue='admitted', data=df,
           size=6,
           fit_reg=False,
           scatter_kws={'s': 50}
          )
plt.show()

# 3. Plot the sigmoid function
# Sigmoid function
sigmoid = lambda z: 1 / (1 + np.exp(-z))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.arange(-10, 10, step=0.01),
        sigmoid(np.arange(-10, 10, step=0.01)))
ax.set_ylim((-0.1,1.1))
ax.set_xlabel('z', fontsize=18)
ax.set_ylabel('g(z)', fontsize=18)
ax.set_title('sigmoid function', fontsize=18)
plt.show()

# 4. Run the logistic regression
X_data = df[['exam1', 'exam2']]
Y_data = df[['admitted']]

numFeatures = X_data.shape[1]

# Tensorflow placeholders for the features and labels data.
X = tf.placeholder(tf.float32, shape=[None, numFeatures])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.zeros([numFeatures, 1]))
b = tf.Variable(tf.zeros([1, 1]))

# Sigmoid is used for the hypotesis - h(x) = x * W + b
pred = tf.nn.sigmoid(tf.matmul(X , W) + b)

cost = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(pred, 1e-9, 1)) + \
                      (1 - Y)*tf.log(tf.clip_by_value(1 - pred, 1e-9, 1)))

correct_predict = tf.equal(tf.cast(tf.greater(pred, 0.5), tf.float32), Y)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
init = tf.initialize_all_variables()

numEpochs = 4000
display_step = 50

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(numEpochs):
        _, cost_value, accuracy_value = sess.run([optimizer, cost, accuracy], feed_dict={X: X_data, Y: Y_data})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print 'Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(cost_value), \
                'accuracy=', accuracy_value

    w_value, b_value = sess.run([W, b], feed_dict={X: X_data, Y: Y_data})
    print w_value, b_value

    # Plot the decision boundary
    sns.set(context='notebook', style='ticks', font_scale=1.5)
    sns.lmplot('exam1', 'exam2', hue='admitted', data=df,
               size=6,
               fit_reg=False,
               scatter_kws={'s': 25}
               )

    plot_x = np.array([X_data.min()[0]-2, X_data.max()[0]+2])
    plot_y = (-1. / w_value[0][0]) * (w_value[1][0] * plot_x + b_value[0][0])
    plt.plot(plot_x, plot_y, 'grey')

    """
    # Alternative plot:
    x = np.arange(130, step=0.1)
    y = -b_value[0][0] / w_value[0][0] - w_value[1][0] / w_value[0][0] * x
    plt.plot(x, y, 'grey')
    """

    plt.xlim(0, 130)
    plt.ylim(0, 130)
    plt.title('Decision Boundary')
    plt.show()
