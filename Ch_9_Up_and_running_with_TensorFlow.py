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

# using with block, no need to close.
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

# easier initialization.
init = tf.initialize_all_variables() # prepare an init node
with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()

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

# Gradient Descent

n_epochs = 1000
learning_rate = 0.01

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)
# => borrowed from page 119.

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# alternative using TF's autodiff:
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)
# alternative using optimizer:
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

# => had to run it manually due to indentation problems.
# => actually, was able to fix it by being careful with indentations in white lines.

# mini-batches and placeholders.

# testing placeholders.
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A+5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)

# now, linear regression with mini-batch gradient descent.
n_epochs = 1000
learning_rate = 0.01

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)
# => borrowed from page 119.

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
xxxx

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# alternative using TF's autodiff:
# gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)
# alternative using optimizer:
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
