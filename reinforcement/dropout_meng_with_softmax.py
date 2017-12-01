import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from laplace_get_data_new2 import read_laplace_data, read_normalized_laplace_data
from sklearn.utils import shuffle
import os

from matplotlib import pyplot as plt
'''
os.makedirs("tmp/model")
os.makedirs("tmp/model-subset")
'''
class Layer():
	def __init__(self, M1, M2, layer_name):
		name = "layer%s" % layer_name
		with tf.name_scope(name):
			self.M1 = M1
			self.M2 = M2
			W = np.random.rand(M1, M2) / np.sqrt(M1+M2)
			b = np.random.rand(M2) / np.sqrt(M2)
			with tf.name_scope('weights'):
				self.W = tf.Variable(W.astype(np.float32))
				tf.summary.histogram(name, self.W)
			with tf.name_scope('biases'):
				self.b = tf.Variable(b.astype(np.float32))
				tf.summary.histogram(name, self.b)
			self.param = [self.W, self.b]

	def forward(self, X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class ANN(object):
	def __init__(self, hidden_layer_size, p_keep):
		self.hidden_layer_size = hidden_layer_size
		self.p_keep = p_keep

	def build_Network(self, X, Y):
		self.hidden_layer = []
		N, D = X.shape
		K = len(set(Y))
		# K = 1
		M1 = D
		count = 1
		for M2 in self.hidden_layer_size:
			h = Layer(M1, M2, count)
			print(M1, M2)
			self.hidden_layer.append(h)
			M1 = M2
			count += 1
		W = np.random.randn(M1, K) / np.sqrt(M1)
		b = np.random.randn(K) / np.sqrt(K)
		with tf.name_scope('last_layer'):
			with tf.name_scope('last_layer/weights'):
				self.W = tf.Variable(W.astype(np.float32))
			with tf.name_scope('last_layer/bioses'):
				self.b = tf.Variable(b.astype(np.float32))

		self.param = [self.W, self.b]
		for h in self.hidden_layer:
			self.param += h.param

	def train(self, X, Y, lr = 1e-4, mu = 0.9, decay = 0.9, epochs = 300, batch_size = 50, split = True, print_every  = 2, wr = False):
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		Y = Y.astype(np.int64)
		#Y = Y.astype(np.float32)
		# if split:
		N, D = X.shape
		K = len(set(Y))
		with tf.name_scope('inputs'):
			inputs = tf.placeholder(tf.float32, shape = (None, D), name = 'inputs')
			#outputs = tf.placeholder(tf.float32, shape = (None,), name = 'outputs')
			labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
		logits = self.forward(inputs)
		prediction = self.predict(inputs)


		with tf.name_scope('cost'):
			# cost = tf.reduce_mean(tf.reduce_sum(tf.square(outputs - prediction), reduction_indices=[1]))  # loss
			# cost = tf.reduce_mean(-tf.reduce_sum(outputs * tf.log(prediction), reduction_indices=[1]))  # loss
			cost = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=logits,
					labels=labels
				)
			)
			tf.summary.scalar('cost', cost)

		with tf.name_scope('error'):
			error_record = np.mean(prediction != Y)
			tf.summary.scalar('error', error_record)


		with tf.name_scope('train'):
			train_op = tf.train.RMSPropOptimizer(lr, decay = decay, momentum = mu).minimize(cost)

		n_batches = int(N / batch_size)

		costs = []
		init = tf.global_variables_initializer()
		
		if wr:
			merged = tf.summary.merge_all()
			write = tf.summary.FileWriter('logs_new/', self.sess.graph)

		saver = tf.train.Saver()

		count = 0
		self.sess.run(init)
		for i in range(epochs):
			print("epoch:", i)
			X, Y = shuffle(X, Y)

			for j in range(n_batches):

				Xbatch = X[j*batch_size : (j+1)*batch_size]
				Ybatch = Y[j*batch_size : (j+1)*batch_size]
				self.sess.run(train_op, feed_dict = {inputs : Xbatch, labels : Ybatch})

				if j % print_every == 0:
					if wr:
						result = self.sess.run(merged, feed_dict = {inputs : Xbatch, labels : Ybatch})
						write.add_summary(result, count)
					c = self.sess.run(cost, feed_dict = {inputs : Xbatch, labels : Ybatch})
					p = self.sess.run(prediction, feed_dict = {inputs : X})
					costs.append(c)
					e = np.mean(p != Y)
					if( i > 399):
						print(p)
					# er = sess.run(error_record, feed_dict = {inputs : X})
					#e = self.error_rate(p, Ybatch)
					print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

				count += 1



		save_path = saver.save(self.sess, "tmp/model/model.ckpt") # 儲存模型到 /tmp/model.ckpt

		#plt.plot(costs)
		#plt.show()
		

		#sess.run(init)

	def error_rate(self, p, Y):
		ans = np.array(abs(p - Y))
		with tf.name_scope('error'):
			err = np.mean(ans < 1)
			tf.summary.scalar('error', err)

		return err

	def set_session(self, session):
		self.sess = session

	def forward(self, X):
        # no need to define different functions for train and predict
        # tf.nn.dropout takes care of the differences for us
		Z = X
		Z = tf.nn.dropout(Z, self.p_keep[0])
		for h, p in zip(self.hidden_layer, self.p_keep[1:]):
			Z = h.forward(Z)
			Z = tf.nn.dropout(Z, p)
		return tf.matmul(Z, self.W) + self.b

	def predict(self, X):
		pY = self.forward(X)
		return tf.argmax(pY, 1)
'''
	def predict(self, X):
		Z = X
		Z = tf.nn.dropout(Z, self.p_keep[0])
		for h, p in zip(self.hidden_layer, self.p_keep[1:]):
			Z = h.forward(Z)
			Z = tf.nn.dropout(Z, p)
		Z = tf.matmul(Z, self.W) +self.b
		return tf.nn.tanh(Z)
'''




def main():
	ann = ANN([40, 150, 100, 80, 45], [0.9, 0.8, 0.9, 0.9, 0.9, 0.9])
	#ann = ANN([40, 150, 100, 80, 40, 20], [0.7, 0.9, 0.8, 0.9, 0.9, 0.9, 0.9])

	#ann = ANN([10, 30, 20, 15], [0.9, 0.9, 0.9, 0.9, 0.9])
	X, Y, _ = read_normalized_laplace_data()
	print(Y)
	Y = Y + 5
	print(Y)
	session = tf.InteractiveSession()
	ann.set_session(session)
	ann.build_Network(X, Y)
	ann.train(X, Y, epochs = 400, wr = True, batch_size = 150)


if __name__ == '__main__':
	main()
