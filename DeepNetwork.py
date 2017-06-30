from tensorflow.examples.tutorials.mnist import input_data as data
from tensorflow.contrib.layers import flatten
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

mnist = data.read_data_sets("MNIST_data/", reshape = False)

X_train, y_train = mnist.train.images, mnist.train.labels
X_valid, y_valid = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

# Make data compatible with LeNet 5 model by padding 2 rows and 2 columns
# on each side. LeNet model accepts 32*32*C images so convert 28*28*C image
# into 32*32 image by padding zeros, here C is the number of color channels

X_train = np.pad(X_train, ((0,0), (2,2), (2,2), (0,0)), 'constant')
X_valid = np.pad(X_valid, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Shuffling the data for better estimation
X_train, y_train = shuffle(X_train, y_train)

# Setting the epoch and batch size parameters for training
EPOCHS = 2200
BATCH_SIZE = 50

# Setting up the LeNet 5 model

# Methods for weight initializations

def weight_variable(shape):
    ''' Form a weight matrix of given shape with gausian intialization
        @parameter: Tuple of required shape
        @returns: Matrix with gausian initialized weights'''
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    ''' Form a bias matrix with a constant value
        @parameter: Shape tuple
        @return: Matrix with given shape and value'''
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# Methods for layer formations

def conv2d(x, W):
    ''' It performs the convolution
        @parameter: Take input matrix and its weight as a input
        @return output weights of the convolution'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    ''' It performs the psubsampling
        @parameter: Take input matrix
        @return: Reduced output matrix'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def fully_connected(x, W):
    ''' Multiply the weights with the input and provides the output
        @parameter: Input Matrix and Weight Matrix
        @return: Output Matrix'''
    return tf.matmul(x, W);

def LeNet5(x):
    ''' This function applies LeNet5 model on the given input
        @parameter: Image matrix
        @return: 10*1 matrix for label estimation'''
    
    # Layer first: Convolution Layer
    W_conv1 = weight_variable([5,5,1,6])
    b_conv1 = bias_variable([6])
    conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    pool1 = max_pool_2x2(conv1)

    # Layer second: Convolution Layer
    W_conv2 = weight_variable([5,5,6,16])
    b_conv2 = bias_variable([16])
    conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
    pool2 = max_pool_2x2(conv2)

    # Roll up the output to form a fully connected layer
    pool2_flat = flatten(pool2)

    # Layer three: Fully connected 400 --> 120
    W_fc1 = weight_variable([400, 120])
    b_fc1 = bias_variable([120])
    fc1 = tf.nn.relu(fully_connected(pool2_flat, W_fc1) + b_fc1)

    # Dropout to make model generalized
    keep_prob = 0.5
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully connected 120 --> 84
    W_fc2 = weight_variable([120, 84])
    b_fc2 = bias_variable([84])
    fc2 = tf.nn.relu(fully_connected(fc1_drop, W_fc2) + b_fc2)

    # Dropout to make model generalized
    keep_prob1 = 0.5
    fc2_drop = tf.nn.dropout(fc2, keep_prob1)

    # Layer 5: Fully connected 84 --> 10
    W_fc3 = weight_variable([84, 10])
    b_fc3 = bias_variable([10])
    fc3 = tf.nn.relu(fully_connected(fc2_drop, W_fc3) + b_fc3)

    return fc3

# Declare the variable to give input of x and y
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10) # to form 10*1 matrix for comparison


# Training Pipeline
rate = 0.0001

logits = LeNet5(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    '''Calculate the accuracy of the test
        @parameter: Take Images Matrix and Label Matrix as Input
        @return: Accuracy'''
    
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train the Model

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    X_train, y_train = shuffle(X_train, y_train)
    
    for i in range(EPOCHS):
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, 'lenet')
    print("Model saved")

# Evaluate the Model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
