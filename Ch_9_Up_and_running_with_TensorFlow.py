# ssh -i ~/Epinomics/alvaroMBPkeypair.pem ubuntu@ec2-52-53-158-144.us-west-1.compute.amazonaws.com
# cd githubRepos/handson_ML_with_SKL_and_TF/
# source activate tensorflow
# python
# ...
# source deactivate

# if using jupyter notebooks:
# ssh -i ~/Epinomics/alvaroMBPkeypair.pem ubuntu@ec2-52-53-158-144.us-west-1.compute.amazonaws.com
# cd githubRepos/handson_ML_with_SKL_and_TF/
# jupyter notebook
# from Firefox: https://52.53.158.144:8888

import tensorflow as tf

# python -c 'import tensorflow; print(tensorflow.__version__)'
# 0.11.0rc0, cool.

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y+y+2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
# 42
sess.close()

# using "with block", no need to close.
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)
# "with block" is similar to a try-catch.

# easier initialization.
init = tf.initialize_all_variables() # prepare an init node
with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()
    print(result)

# x is calculated twice, when needed for y and when needed for z.
w = tf.constant(3)
x = w+2
y = x+5
z = x*3
with tf.Session() as sess:
    print(y.eval()) # 10
    print(z.eval()) # 15

# how to calculate x only once.
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val) # 10
    print(z_val) # 15

# Linear regression with TensorFlow

# need to
# conda install matplotlib pandas scipy scikit-learn

# Using the formula/normal equation.
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

tf.reset_default_graph()
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

print(theta_value)
[[ -3.74636536e+01]
 [  4.35707688e-01]
 [  9.34202131e-03]
 [ -1.06593400e-01]
 [  6.43945396e-01]
 [ -4.25595226e-06]
 [ -3.77299148e-03]
 [ -4.26701456e-01]
 [ -4.40539479e-01]]

# Gradient Descent: manually computing gradients.

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
# => borrowed from page 119.
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
print(scaled_housing_data_plus_bias.mean(axis=0))

tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)
[[ 0.90454292]
 [ 0.35481548]
 [ 0.59063649]
 [ 0.51156354]
 [-0.04808879]
 [ 0.26202965]
 [-0.62795925]
 [-0.77138448]
 [-0.32755637]]

# => had to run it manually due to indentation problems.
# => actually, was able to fix it by being careful with indentations in white lines.
# => however, it prints the thetas, as opposed to the MSEs; couldn't fix it.

# using autodiff
# Same as above except for the gradients = ... line.
# basically TF is calculating the gradients for me.
tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = tf.gradients(mse, [theta])[0]
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)
[[  2.06855249e+00]
 [  7.74078071e-01]
 [  1.31192416e-01]
 [ -1.17845096e-01]
 [  1.64778158e-01]
 [  7.44091696e-04]
 [ -3.91945094e-02]
 [ -8.61356437e-01]
 [ -8.23479593e-01]]

# using an optimizer (GradientDescentOptimizer)
# replace the gradients = ... and training_op = ... lines
tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

# => for some reason this one DOES print the MSEs!
print("Best theta:")
print(best_theta)
[[  2.06855249e+00]
 [  7.74078071e-01]
 [  1.31192416e-01]
 [ -1.17845096e-01]
 [  1.64778158e-01]
 [  7.44091696e-04]
 [ -3.91945094e-02]
 [ -8.61356437e-01]
 [ -8.23479593e-01]]

# mini-batches and placeholders.
# placeholder nodes are special because they don’t actually perform any computation,
# they just output the data you tell them to output at runtime. If you don’t specify
# a value at runtime for a placeholder, you get an exception.

# testing placeholders.
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A+5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
print(B_val_2)

# now, linear regression with mini-batch gradient descent.
tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.initialize_all_variables()

import numpy.random as rnd
def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)
    indices = rnd.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

print("Best theta:")
print(best_theta)
[[ 2.06697559]
 [ 0.82894456]
 [ 0.11803052]
 [-0.23456885]
 [ 0.29808956]
 [ 0.00392114]
 [-0.00724683]
 [-0.90761697]
 [-0.88751072]]

# Saving and restoring models
tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
            save_path = saver.save(sess, "my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, "my_model_final.ckpt")

print("Best theta:")
print(best_theta)
[[  2.06855249e+00]
 [  7.74078071e-01]
 [  1.31192416e-01]
 [ -1.17845096e-01]
 [  1.64778158e-01]
 [  7.44091696e-04]
 [ -3.91945094e-02]
 [ -8.61356437e-01]
 [ -8.23479593e-01]]

# Visualizing the graph and training curves using TensorBoard

tf.reset_default_graph()

from datetime import datetime

# need to create a log dir for TensorBoard.
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
# removing the seed so that I get different initializations:
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.initialize_all_variables()

# A summary to keep track of stats flowing through the graph.
mse_summary = tf.scalar_summary('MSE', mse)
summary_writer = tf.train.SummaryWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            # create summary every 10 batches.
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

summary_writer.flush()
summary_writer.close()
print("Best theta:")
print(best_theta)
[[ 2.06697559]
 [ 0.82894456]
 [ 0.11803052]
 [-0.23456885]
 [ 0.29808956]
 [ 0.00392114]
 [-0.00724683]
 [-0.90761697]
 [-0.88751072]]
# try to recreate the two initial directories with different
# initializations.
# => did that, works well, maybe report to author, maybe on GitHub?

# TensorBoard
# had to open TCP port 6006 on instance.
cd /home/ubuntu/githubRepos/handson_ML_with_SKL_and_TF
# => where the tf_logs directory was created.
source activate tensorflow
tensorboard --logdir tf_logs
# => notice it tells me an incorrect IP address to use.
http://52.53.158.144:6006
# => chrome works well.
# was left running within screen "tensorboard"

# Using name scopes to unclutter visualization of network nodes.
tf.reset_default_graph()

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope('loss') as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.initialize_all_variables()

mse_summary = tf.scalar_summary('MSE', mse)
summary_writer = tf.train.SummaryWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

summary_writer.flush()
summary_writer.close()
print("Best theta:")
print(best_theta)
# looks great!

# Modularity.

# ugly flat code.
tf.reset_default_graph()

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

linear1 = tf.add(tf.matmul(X, w1), b1, name="linear1")
linear2 = tf.add(tf.matmul(X, w2), b2, name="linear2")

relu1 = tf.maximum(linear1, 0, name="relu1")
# relu2 = tf.maximum(linear1, 0, name="relu2")  # Oops, cut&paste error! Did you spot it?
relu2 = tf.maximum(linear2, 0, name="relu2")  # Error spotted and fixed :-D

output = tf.add_n([relu1, relu2], name="output")

# better, use a function to create the ReLU.
tf.reset_default_graph()

def relu(X):
    w_shape = int(X.get_shape()[1]), 1
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    linear = tf.add(tf.matmul(X, w), b, name="linear")
    return tf.maximum(linear, 0, name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
summary_writer = tf.train.SummaryWriter("tf_logs/relu1", tf.get_default_graph())
summary_writer.close()

# even better, using name scopes.
tf.reset_default_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = int(X.get_shape()[1]), 1
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        linear = tf.add(tf.matmul(X, w), b, name="linear")
        return tf.maximum(linear, 0, name="max")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

summary_writer = tf.train.SummaryWriter("tf_logs/relu2", tf.get_default_graph())
summary_writer.close()
# check GRAPHS in TensorBoard.
# beautiful.

xxxx next, sharing variables.
