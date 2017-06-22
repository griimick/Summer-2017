# BASIC LOGIC FOR A MODEL
# def model_fn(features, targets, mode, params):
#    # Logic to do the following:
#    # 1. Configure the model via TensorFlow operations
#    # 2. Define the loss function for training/evaluation
#    # 3. Define the training operation/optimizer
#    # 4. Generate predictions
#    # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
#    return ModelFnOps(mode, predictions, loss, train_op, eval_metric_ops)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # DataSet images are 612x452 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 612, 452, 1])

  # Convolutional Layer #1
  # Computes 320 features using a 20x20 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 612, 452, 1]
  # Output Tensor Shape: [batch_size, 612, 452, 320]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=320,
      kernel_size=[20, 20],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 612, 452, 320]
  # Output Tensor Shape: [batch_size, 306, 226, 320]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 640 features using a 20x20 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 306, 226, 320]
  # Output Tensor Shape: [batch_size, 306, 226, 640]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=640,
      kernel_size=[20, 20],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 306, 226, 640]
  # Output Tensor Shape: [batch_size, 153, 113, 640]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3
  # Computes 640 features using a 20x20 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 153, 113, 640]
  # Output Tensor Shape: [batch_size, 153, 113, 640]
  conv3 = tf.layers.conv2d(
      inputs=pool1,
      filters=640,
      kernel_size=[20, 20],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 153, 113, 640]
  # Output Tensor Shape: [batch_size, 76, 56, 640]
  pool3 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #4
  # Computes 640 features using a 20x20 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 76, 56, 640]
  # Output Tensor Shape: [batch_size, 76, 56, 320]
  conv4 = tf.layers.conv2d(
      inputs=pool1,
      filters=320,
      kernel_size=[20, 20],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #4
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 76, 56, 320]
  # Output Tensor Shape: [batch_size, 38, 28, 320]
  pool4 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #5
  # Computes 640 features using a 10x10 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 38, 28, 320]
  # Output Tensor Shape: [batch_size, 38, 28, 320]
  conv5 = tf.layers.conv2d(
      inputs=pool1,
      filters=320,
      kernel_size=[10, 10],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #5
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 38, 28, 320]
  # Output Tensor Shape: [batch_size, 19, 14, 320]
  pool5 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #6
  # Computes 640 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 19, 14, 320]
  # Output Tensor Shape: [batch_size, 19, 14, 160]
  conv6 = tf.layers.conv2d(
      inputs=pool1,
      filters=160,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #6
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 19, 14, 160]
  # Output Tensor Shape: [batch_size, 9, 7, 160]
  pool6 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 9, 7, 160]
  # Output Tensor Shape: [batch_size, 9 * 7 * 160]
  pool6_flat = tf.reshape(pool6, [-1, 9 * 7 * 160])

  # Dense Layer #1
  # Densely connected layer with 5040 neurons
  # Input Tensor Shape: [batch_size, 9 * 7 * 160]
  # Output Tensor Shape: [batch_size, 5040]
  dense1 = tf.layers.dense(inputs=pool6_flat, units=5040, activation=tf.nn.relu)

  # Dense Layer #2
  # Densely connected layer with 2520 neurons
  # Input Tensor Shape: [batch_size, 5040]
  # Output Tensor Shape: [batch_size, 2520]
  dense2 = tf.layers.dense(inputs=dense1, units=2520, activation=tf.nn.relu)

  # Dense Layer #3
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 2520]
  # Output Tensor Shape: [batch_size, 1024]
  dense3 = tf.layers.dense(inputs=dense2, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense3, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 2]
  logits = tf.layers.dense(inputs=dropout, units=2)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

height=612
width=452
nClass=2

# Function to tell TensorFlow how to read a single image from input file
def getImage(filename):
    # convert filenames to a queue for an input pipeline.
    filenameQ = tf.train.string_input_producer([filename],num_epochs=None)
 
    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example 
    key, fullExample = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/channels':  tf.FixedLenFeature([], tf.int64),            
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })


    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg',[image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)
    
        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel. 
    # the "1-.." part inverts the image, so that the background is black.

    image=tf.reshape(1-tf.image.rgb_to_grayscale(image),[height*width])

    # re-define label as a "one-hot" vector 
    # it will be [0,1] or [1,0] here. 
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, nClass))

    return label, image

def main(unused_argv):
  # Load training and eval data
  mnist = learn.datasets.load_dataset("mnist")
  print(mnist)
  # train_data = mnist.train.images  # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  train_labels, train_data = getImage("train-00000-of-00001")
  
  # train_data = 
  # train_labels =  
  # # eval_data =
  # eval_labels = 

  # Create the Estimator
  data_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=10)

  # Train the model
  learn.SKCompat(data_classifier).fit(
      x=train_data,
      y=train_labels,
      batch_size=1,
      steps=21,
      monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
      "accuracy":
          learn.MetricSpec(
              metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  # eval_results = data_classifier.evaluate(
  #     x=eval_data, y=eval_labels, metrics=metrics)
  # print(eval_results)


if __name__ == "__main__":
  tf.app.run()