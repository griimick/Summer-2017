#Simple Nand Gate Neural Net
import tensorflow as tf
import numpy

# Train Data
train_X = numpy.asarray([[0,0],[0,1],[1,0],[1,1]])
# Expected Labels
train_Y = numpy.asarray([[1],[1],[1],[0]])

# Placeholder for Tensors
x = tf.placeholder(tf.float32,[None, 2])
y = tf.placeholder(tf.float32,[None, 1])

# Variables for Weighs and Baises
weights = tf.Variable(tf.zeros([2, 1]))
biases = tf.Variable(tf.zeros([1, 1]))

# Activation Function, here Sigmoid Function 
activation = tf.nn.sigmoid(tf.matmul(x, weights)+biases)

# Cost Function 
cost = tf.reduce_sum(tf.square(activation - y))/4

# Optimizing Function, here GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(.1).minimize(cost)

init = tf.global_variables_initializer()

# Number of Epochs
epoch = 5000

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        epoch_loss = 0
        train_data, c = sess.run([optimizer,cost], feed_dict={x: train_X, y: train_Y})
        epoch_loss += c

    try_X = numpy.asarray([[0,1]])
    result = sess.run(activation, feed_dict={x:try_X})
    # Rounding off results
    # for r in range(4):
    # 	if (result[r][0]>0.5):
    # 		result[r][0] = 1
    # 	else:
    # 		result[r][0] = 0


    print(result[0][0])
    print("Loss ", epoch_loss)

    