import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from laplace_get_data_new2 import read_laplace_data, read_normalized_laplace_data
from sklearn.utils import shuffle
#from matplotlib import pyplot as plt

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
				#tf.summary.histogram(name, self.W)
			with tf.name_scope('biases'):
				self.b = tf.Variable(b.astype(np.float32))
				#tf.summary.histogram(name, self.b)
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
		self.D = D
		self.i = tf.placeholder(tf.float32, shape = (None, D), name = 'inputs')

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
		b = np.zeros(K)
		with tf.name_scope('last_layer'):
			with tf.name_scope('last_layer/weights'):
				self.W = tf.Variable(W.astype(np.float32))
			with tf.name_scope('last_layer/bioses'):
				self.b = tf.Variable(b.astype(np.float32))

		self.param = [self.W, self.b]
		for h in self.hidden_layer:
			self.param += h.param

		saver = tf.train.Saver()

		saver.restore(self.sess, "tmp/model/model.ckpt")



	def weights_biases_output(self):
		weights_output = []
		biases_output = []
		for layer in self.hidden_layer:
			weights_output.append(self.sess.run(self.layer.W))
			biases_output.append(self.sess.run(self.layer.b))
		wo = pd.DataFrame()

	def predict(self, X):

		Z = X
		Z = tf.nn.dropout(Z, self.p_keep[0])
		for h, p in zip(self.hidden_layer, self.p_keep[1:]):
			Z = h.forward(Z)
			Z = tf.nn.dropout(Z, p)
		Z = tf.matmul(Z, self.W) + self.b
		#return Z
		return tf.argmax(Z, axis = 1)


	def predict_out(self, X):

		X = np.atleast_2d(X)
		prediction_o = self.predict(self.i)
		out = self.sess.run(prediction_o, feed_dict = {self.i : X})
		return out


	def set_session(self, session):
		self.sess = session
'''

	def predict(self, X):
		pY = self.forward(X)
		return tf.argmax(pY, 1)
'''

def create_NN(session):
	ann = ANN([40, 150, 100, 80, 45], [0.9, 0.8, 0.9, 0.9, 0.9, 0.9])
	X, Y, _ = read_normalized_laplace_data()
	ann.set_session(session)
	ann.build_Network(X, Y)
	return ann

def To_normalize_data(input_of_NN, normalize_data):
    input_of_NN_normalize = []
    normalize_data = np.array(normalize_data)
    normalize_data = np.log(normalize_data + 1)
    for i, n in zip(input_of_NN, normalize_data):
        input_of_NN_normalize.append((i - n[0]) / n[1])
    return input_of_NN_normalize

def generate_input(o, mn, mm):
	return [[o, o**2, o**3, o**mn, mn*o, mn, mn**2, mn**3, mn, mm**mn, mm**2, mm**o, mm*o*mn, o**mm]]


def main():
	ann = ANN([40, 150, 100, 80, 45], [0.9, 0.8, 0.9, 0.9, 0.9, 0.9])
	X, Y, normalize_data = read_normalized_laplace_data()
	session = tf.InteractiveSession()
	ann.set_session(session)
	ann.build_Network(X, Y)
	N, D = X.shape
	# ann.train(X, Y, epochs = 300, wr = True, batch_size = 150)
	o = 1
	mn = 2
	mm = 3
	input_of_NN = generate_input(o, mn, mm)
	for i in range(N):
		#print(To_normalize_data(X[i], normalize_data))
		print(ann.predict_out(X[i]), Y[i] + 5)
		#print(tf.argmax(ann.predict_out(To_normalize_data(X[i*100], normalize_data)), axis = 1))
		#print("\n\n")



if __name__ == '__main__':
	main()