import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def add_layer(data, in_size, out_size, activation=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size])+0.1)
	Wx_plus_b = tf.matmul(data, Weights) + biases
	if activation == None:
		outputs = Wx_plus_b
	else:
		outputs = activation(Wx_plus_b)
	return outputs

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs: v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
	return result

#define placeholder for inputs to network

xs = tf.placeholder(tf.float32, [None, 784]) #28*28
ys = tf.placeholder(tf.float32, [None, 10]) 
prediction = add_layer(xs, 784, 10, activation=tf.nn.softmax)

# cross entropy for neural network
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
	if i%50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))













