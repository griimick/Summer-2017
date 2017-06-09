import tensorflow as tf
import numpy

train_X = numpy.asarray([[0,0],[0,1],[1,0],[1,1]])
train_Y = numpy.asarray([[1],[1],[1],[0]])

x = tf.placeholder("float",[None, 2])
y = tf.placeholder("float",[None, 1])

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1, 1]))

activation = tf.nn.sigmoid(tf.matmul(x, W)+b)
cost = tf.reduce_sum(tf.square(activation - y))/4
optimizer = tf.train.GradientDescentOptimizer(.1).minimize(cost)

init = tf.initialize_all_variables()

epoch = 5000

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        train_data = sess.run(optimizer, feed_dict={x: train_X, y: train_Y})

    result = sess.run(activation, feed_dict={x:train_X})
    print(result)