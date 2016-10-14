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

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

init = tf.initialize_all_variables() # prepare an init node 
with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()
