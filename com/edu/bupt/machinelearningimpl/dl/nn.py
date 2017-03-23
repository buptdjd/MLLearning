import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation==None:
		outputs = Wx_plus_b
	else:
		outputs = activation(Wx_plus_b)
	return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis].astype(np.float32)
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None, 1])
ys = tf.placeholder(tf.float32,[None, 1])

h1 = add_layer(xs, 1, 10, activation=tf.nn.relu)
p1 = add_layer(h1, 10, 1, None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(p1-ys), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
#plt.show(block=False)

for step in range(1000):
	sess.run(train_step, feed_dict={xs: x_data, ys:y_data})
	if step%50 == 0:
		try:
			ax.lines.remove(lines[0])
		except Exception:
			pass
		#print(sess.run(loss, feed_dict={xs: x_data, ys:y_data}))
		p = sess.run(p1, feed_dict = {xs: x_data})
		lines = ax.plot(x_data, p, 'r', lw=5)
		plt.pause(0.1)
#plt.scatter(x_data, y_data)
#plt.show()



