import tensorflow as tf
import numpy

train_X = numpy.asarray([[0,0],[0,1],[1,0],[1,1]])
train_Y = numpy.asarray([[1],[1],[1],[0]])

x = tf.placeholder("float",[None, 2])
y = tf.placeholder("float",[None, 1])

weights = tf.Variable(tf.zeros([2, 1]))
biases = tf.Variable(tf.zeros([1, 1]))

activation = tf.nn.sigmoid(tf.matmul(x, weights)+biases)
cost = tf.reduce_sum(tf.square(activation - y))/4
optimizer = tf.train.GradientDescentOptimizer(.1).minimize(cost)

init = tf.initialize_all_variables()

epoch = 5000

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        epoch_loss = 0
        train_data = sess.run(optimizer, feed_dict={x: train_X, y: train_Y})
        _, c = sess.run([optimizer, cost], feed_dict={x: train_X, y: train_Y})
        epoch_loss += c

    result = sess.run(activation, feed_dict={x:train_X})

    # Rounding off results
    for r in range(4):
    	if (result[r][0]>0.5):
    		result[r][0] = 1
    	else:
    		result[r][0] = 0

    print(result)
    print("Loss ", epoch_loss)