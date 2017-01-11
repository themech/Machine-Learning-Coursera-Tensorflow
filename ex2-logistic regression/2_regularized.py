import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

FEATURE_MAPPING_POWER = 6
NUM_EPOCHS = 25000
REG_LAMBDA = 0.1 # overfit: 0.0001, ok: 0.1-0.01, underfit: 5
MESH_RESOLUTION = 250.0

# 1. Read the data and print the sample
df = pd.read_csv('data/ex2data2.txt', names=['test1', 'test2', 'accepted'])
print(df.shape)
print df.head()

# 2. Visualize the data
sns.set(context='notebook', style='darkgrid', palette=sns.color_palette('RdBu', 2))
sns.lmplot('test1', 'test2', hue='accepted', data=df,
           size=6,
           fit_reg=False,
           scatter_kws={'s': 50}
          )

plt.show()

# 3. Run the logistic regression with feature mapping and regularization


def feature_mapping(f1, f2, power):
    """Helper function for feature mapping"""
    data = {'f{}{}'.format(i - p, p): np.power(f1, i - p) * np.power(f2, p)
                for i in np.arange(power + 1)
                for p in np.arange(i + 1)
            }

    return pd.DataFrame(data)

X_data = feature_mapping(df.test1, df.test2, power=FEATURE_MAPPING_POWER)
Y_data = df[['accepted']]
numFeatures = X_data.shape[1]
numSamples = X_data.shape[0]

# Tensorflow placeholders for the features and labels data.
X = tf.placeholder(tf.float32, shape=[None, numFeatures])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.zeros([numFeatures, 1]))
# after feature mapping, f00 is always 1, so W[0] will be our b

# Sigmoid is used for the hypotesis - h(x) = x * W + b
pred = tf.nn.sigmoid(tf.matmul(X , W))

cost = -tf.reduce_sum(Y * tf.log(tf.clip_by_value(pred, 1e-9, 1)) + \
                      (1 - Y)*tf.log(tf.clip_by_value(1 - pred, 1e-9, 1))) / numSamples
regularized_W = tf.slice(W, [1,0], [-1,-1]) # don't regularize W[0]
regularizer = tf.reduce_sum(tf.square(regularized_W)) * REG_LAMBDA / numFeatures

correct_predict = tf.equal(tf.cast(tf.greater(pred, 0.5), tf.float32), Y)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

optimizer = tf.train.AdamOptimizer(0.01).minimize(cost + regularizer)
init = tf.initialize_all_variables()

display_step = 50

with tf.Session() as sess:
    sess.run(init)
    w_value = []
    for epoch in range(NUM_EPOCHS):
        _, cost_value, reg_cost, accuracy_value, w_value = sess.run([optimizer, cost, regularizer, accuracy, W], feed_dict={X: X_data, Y: Y_data})
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print 'Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(cost_value), 'reg=', '{:.9f}'.format(reg_cost), \
                'accuracy=', accuracy_value

    print 'w_value', w_value

    # Plot the input data
    x_pos = [v for v in df['test1'].values]
    y_pos = [v for v in df['test2'].values]
    labels = ['r' if v == 1 else 'b' for v in Y_data.values]
    plt.scatter(x_pos, y_pos, c=labels, edgecolor='k', s=50)

    # Plot the decision boundary contour
    x_min, x_max = min(x_pos) - 0.5, max(x_pos) + 0.5
    y_min, y_max = min(y_pos) - 0.5, max(y_pos) + 0.5
    mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / MESH_RESOLUTION),
                                 np.arange(y_min, y_max, (y_max - y_min) / MESH_RESOLUTION))
    pts = feature_mapping(mesh_x.ravel(), mesh_y.ravel(), power=FEATURE_MAPPING_POWER)
    classifier = tf.greater(pred, 0.5)
    mesh_color = sess.run(classifier, feed_dict={X: pts})
    mesh_color = np.array(mesh_color).reshape(mesh_x.shape)
    plt.contour(mesh_x, mesh_y, mesh_color, linewidths=2)

    plt.show()
