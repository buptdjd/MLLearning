import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


def add_layer(data, in_size, out_size, layer_name, activation=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(data, Weights) + biases
	Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
	if activation == None:
		outputs = Wx_plus_b
	else:
		outputs = activation(Wx_plus_b)
	tf.summary.histogram(layer_name + "/ouputs", outputs)
	return outputs

# placeholder
keep_prob = tf.placeholder(tf.float32, None)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

# add layer
h1 = add_layer(xs, 64, 100, "h1", activation = tf.nn.tanh)

pred = add_layer(h1, 100, 10, "h2", activation = tf.nn.softmax)

init = tf.global_variables_initializer()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred),reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

merged = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

sess.run(init)


for i in range(500):
	sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
	if i%50 == 0:
		train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob:1})
		test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob:1})
		train_writer.add_summary(train_result, i)
		test_writer.add_summary(test_result, i)
	

