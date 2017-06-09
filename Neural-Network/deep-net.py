# Following Neural Netwrok Model - Deep Learning With Neural Netwrok and TenserFlow
# https://www.youtube.com/watch?v=BhpvH5DuVu8
# By Harrison Kinsley (sentdex)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Number of nodes in each hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Number of final classes
n_classes = 10
batch_size = 100

# tf.placeholder acts as target for the feed, will generate error if not fed
# tf.placeholder takes in datatype, shape (optional) and name(of operation, optional) as input 
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	# Each hidden layer is a dictionary consisting 'weights' and 'biases'
	# Size of weights in a layer is input size x the number of nodes in that layer
	# Size of biases of a given layer is equal to the number of nodes in that layer
	# (inputdata * weights) + biases
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

    # (inputdata * weights) + biases
    #ACTIVATION FUNCTION
    # nn.relu() is the activation / threshold function 
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    #TODO: Why is this different? not using tf.add but simple addition using operator
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

# Continuation of the second video of the tutorial : https://youtu.be/PwAGxqrXSCs
def train_neural_network(x):

	# Predictions using the Model that was modelled above
    prediction = neural_network_model(x)

    #COST FUNCTION
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )

    # deafult learning rate  = 0.001 in AdamOptimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:

        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # Running in batches
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)


