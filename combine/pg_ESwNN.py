import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from get_laplace_data import read_laplace_data, read_normalized_laplace_data

#####################################  notice  ###############################
#  This file use ESwNN's data to initailize the network of pg.
#  This file is the first version of combination.
#  In this file, I just use one Policy-Gradient network to be the brain of agent
#  the next step I'm going to do is seperate the network to 2 or more network
#  for trainning different agent to face different situations.




############initial setting#############
n = 128
ep_max_step = 100000
file_name = "pg_random.txt"
_, _, normalize_data = read_normalized_laplace_data()


def check_save_folder_exist():
	file_list = os.listdir()
	if not 'tmp' in file_list:
		os.makedirs("tmp/model")
		os.makedirs("tmp/model-subset")

def To_normalize_data(input_of_NN, normalize_data):
    input_of_NN_normalize = []
    normalize_data = np.array(normalize_data)
    normalize_data = np.log(normalize_data + 1)
    for i, n in zip(input_of_NN, normalize_data):
        input_of_NN_normalize.append((i - n[0]) / n[1])
    return input_of_NN_normalize

def transfrom_input_of_NN( err, r):
    if (err == 0):
            o = 19
    else:
        o = int(-np.ceil(np.log10(err)))

    if ( o <=0):
        o =1
    m = r    
    mn = n / 100
    mm = (m+1) / 10
        
    #input_of_NN = [o, o**2, o**3, o**mn, mn*o, mn, mn**2, mn**3, mn, mm**mn, mm**2, mm**o, mm*o*mn, o**mm]
    input_of_NN = [o, o**2, o**3, o**mn, mn**o, mn**2, mn**3, mn, mm**mn, mm**2, mm**o, mm*o*mn, o**mm]
    return To_normalize_data(input_of_NN, normalize_data)


class HiddenLayer:
	def __init__(self, M1 = 0, M2 = 0, f = tf.nn.tanh, use_bias = True, zeros = False, copy_from_ESwNN = False, W_E = 0, b_E = 0):
		self.use_bias = use_bias
		if not copy_from_ESwNN:
			if zeros:
				W = np.zeros((M1, M2)).astype(np.float32)
			else:
				W = np.random_normal(shape = (M1, M2))
			if use_bias:
				b = np.zeros(M2).astype(np.float32)
		else:
			W = np.array(W_E).astype(np.float32)
			b = np.array(b_E).astype(np.float32)

		self.W = tf.Variable(W)
		self.b = tf.Variable(b)
		self.params = [self.W, self.b]

		self.f = f

	def forward(self, X):
		if self.use_bias:
			output = tf.matmul(X, self.W) + self.b
		else:
			output = tf.matmul(X, self.W)
		return self.f(output)


class PolicyModel:
	def __init__(self, D, net_params, net_shapes, session = None, save_network = False):
		self.D = D
		K = 10
		self.net_shapes = net_shapes
		self.net_params = net_params
		self.layers = []
		self.params = []
		self.save_network = save_network
		self.set_session(session)

		########### copy the data from the ESwNN ###############
		start = 0
		for i, shape in enumerate(self.net_shapes):
			n_w, n_b = shape[0] * shape[1], shape[1]
			W_E, b_E = self.net_params[start: start + n_w].reshape(shape), self.net_params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))
			layer = HiddenLayer(copy_from_ESwNN = True, W_E = W_E, b_E = b_E)
			self.layers.append(layer)
			start += n_w + n_b

		print("start to save\n\n")
		############### record the network ##################
		for layer in (self.layers):
			self.params += layer.params

		self.save_or_load()
		#self.show_network()


		############### set some tensorflow term ############
		self.X = tf.placeholder(tf.float32, shape = (None, self.D), name = 'X')
		self.actions = tf.placeholder(tf.int32, shape = (None,), name = 'actions')
		self.advantages = tf.placeholder(tf.float32, shape = (None,), name = 'advantages')

		Z = self.X
		for layer in self.layers:
			Z = layer.forward(Z)
		p_a_given_s = Z

		self.predict_op = p_a_given_s

		selected_probs = tf.log(tf.reduce_sum(p_a_given_s * tf.one_hot(self.actions, K), reduction_indices = [1]))

		cost = -tf.reduce_sum(self.advantages * selected_probs)

		self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)


	def save_or_load(self):
		init_op = tf.global_variables_initializer()
		self.session.run(init_op)
		print("init\n\n")

		########### some setting to get/save the network ############
		saver = tf.train.Saver()
		if(self.save_network):
			check_save_folder_exist()
			save_path = saver.save(self.session, "tmp/model/model.ckpt")

		else:
			saver.restore(self.session, "tmp/model/model.ckpt")
 

	def show_network(self):
		for layer in self.layers:
			print(self.session.run(layer.W))
			print("\n\n\n\n")
			print(self.session.run(layer.b))
			print("\n\n\n\n\n")


	def set_session(self, session):
		self.session = session

	def init_vars(self):
		init_op = tf.variables_initializer(self.params)
		self.session.run(init_op)

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict = {self.X: X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return np.random.choice(len(p), p = p)

	def perturb_params(self):
		ops = []
		for p in self.params:
			v = self.session.run(p)
			noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
			if (np.random.random() < 0.1):
				op = p.assign(noise)
			else:
				op = p.assign(v + noise)
			ops.append(op)

		self.session.run(ops)

	def partial_fit(self, X, actions, advantages):
		X = np.atleast_2d(X)
		actions = np.atleast_1d(actions)
		advantages = np.atleast_1d(advantages)
		self.session.run(self.train_op, feed_dict = {self.X : X, self.actions : actions, self.advantages : advantages})


class ValueModel:
	def __init__(self, D, hidden_layer_size):
		self.layers = []
		K = 10

		########### build the network ############
		M1 = D
		for M2 in hidden_layer_size:
			layer = HiddenLayer(M1, M2)
			self.layers.append(layer)
			M1 = M2
		layer = HiddenLayer(M1, 1)
		self.layers.append(layer)

		########### tensorflow setting ###########
		self.X = tf.placeholder(tf.float32, shape = (None, D), name = 'value_X')
		self.Y = tf.placeholder(tf.float32, shape = (None, 1), name = 'Y')

		Z = self.X
		for layer in layers:
			Z = layer.forward(Z)
		Y_hat = tf.reshape(Z, [-1])
		self.predict_op = Y_hat

		cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
		self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)

	def set_session(self, session):
		self.session = session

	def partial_fit(self, X, Y):
		X = np.atleast_2d(X)
		Y = np.atleast_1d(Y)
		self.session.run(self.train_op, feed_dict = { self.X:X, self.Y:Y })

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict = { self.X:X })




def play_one(pmodel, vmodel, gamma, line):

	############## gpu setting ####################

	T = np.zeros((n,n))
	T[:,0] = 10

	T = T.astype(np.float64)
	err = np.zeros_like(T.flatten())

	_err = cuda.mem_alloc(T.nbytes)
	T1 = cuda.mem_alloc(T.nbytes)
	T2 = cuda.mem_alloc(T.nbytes)
	cuda.memcpy_htod(T1,T)
	cuda.memcpy_htod(T2,T)


	mod = SourceModule("""
    __global__ void Laplace(double *T_old, double *T_new, double *err, double ratio, int n)
    {
        // compute the "i" and "j" location of the node point
        // handled by this thread

        int i = blockIdx.x * blockDim.x + threadIdx.x ;
        int j = blockIdx.y * blockDim.y + threadIdx.y ;

        // get the natural index values of node (i,j) and its neighboring nodes
                                    //                         N 
        int P = i + j*n;           // node (i,j)              |
        int N = i + (j+1)*n;       // node (i,j+1)            |
        int S = i + (j-1)*n;       // node (i,j-1)     W ---- P ---- E
        int E = (i+1) + j*n;       // node (i+1,j)            |
        int W = (i-1) + j*n;       // node (i-1,j)            |
                                    //                         S 

        // only update "interior" node points
        if(i>0 && i<n-1 && j>0 && j<n-1) {
            T_new[P] = 0.25*( T_old[E] + T_old[W] + T_old[N] + T_old[S] );
            T_new[P] = ratio*T_new[P] + (1-ratio)*T_old[P];
        }
        __syncthreads();
        err[P] = abs( (T_new[P] - T_old[P]));
    }

      """)

	bdim = (16, 16, 1)
	gdim = (8,8,1)
	func = mod.get_function("Laplace")

    ############## create the first observation ##################### 
	ratio = 1.0
	for i in range(3):
		func(T1, T2, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
		func(T2, T1, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
	cuda.memcpy_dtoh(err, _err)
	max_err = max(err)
	observation = transfrom_input_of_NN(max_err, r = ratio)
	ep_r = 0

	# set some variable to improve the learning step
	# i am going to store 3 error
	err_1 = max_err
	err_2 = 10

	############## solve by the network ######################### 

	states = []
	actions = []
	ep_r = 0

	for step in range(ep_max_step):
		action = pmodel.sample_action(observation)
		states.append(observation)
		actoins.append(action)

		for i in range(1000):
			func(T1, T2, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
			func(T2, T1, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
		cuda.memcpy_dtoh(err, _err)
        
		max_err = max(err)
		observation = transfrom_input_of_NN(max_err, r = ratio)           

		# I use the eps times to be the reward
		# and the network is going to update when 
		# the one episode is finished.
		if (err_1 < max_err and err_2 < err_1):
			ep_r -= 20000
		ep_r -= 3

		if max_err < 10e-16: break

	############## train the policy and value network ################

	# In the typical RL model, it is gonna output a reward every single step.
	# Therefore, in this case, as we dont have a enivorment to see how the reinforcemnet
	# the agent get, we can only see the output at the last of every episode.
	# If we use decay factor, as this case is a long game, the rewards of first half's steps can get 
	# would be a small value. Therefore, I'm not going to use decay factor here. It means that
	# every steps of a single episode would get the same reward.

	# line is the data get from ESwNN,
	# reward is between -2 and 2
	reward = ( 2 * (ep_r - line[1]) / (line[0] - line[1]) -1 ) *2

	# knowledge:
	# advantages represents the update's direction
	# It means that if advantage is small or negative,
	# the direction of gradient would be a negative,
	# the network would think it is a bad action in this situation.

	returns = []
	advantages = []
	G = 0
	for s in reversed(states):
		returns.append(G)
		advantages.append(G - vmodel.predict(s)[0])
		G = rewards + gamma*G
	returns.reverse()
	advantages.reverse()

	pmodel.partial_fit(states, actions, advantages)
	vmodel.partial_fit(states, returns)

	return ep_r


def play_multiple_episodes(times, pmodel, gamma, print_iters=False):
	totaleps = np.empty(times)

	for i in range(times):
		totaleps[i] = play_one(pmodel, gamma)

		if print_iters:
			print(i, "avg so far:", totaleps[:(i+1)].mean())

	avg_totaleps = totaleps.means()
	print("avg total eps:", avg_totaleps)
	return avg_totaleps

def random_search(pmodel, gamma):
	totaleps = []
	best_avg_totaleps = float('-inf')
	best_pmodel = pmodel
	num_episodes_per_param_test = 3
	for t in range(10):
		tmp_pmodel = best_pmodel.copy()
		tmp_pmodel.perturb_params()
		avg_totaleps = play_multiple_episodes(num_episodes_per_param_test, tmp_pmodel, gamma)
		totaleps.append(avg_totaleps)

		if(avg_totaleps > best_avg_totaleps):
			best_avg_totaleps = avg_totaleps
			best_pmodel = tmp_pmodel

	return totaleps, best_pmodel

def train(pmodel, vmodel, gamma, line):
	train_time = 60
	costs = np.empty(train_time)
	iters = np.empty(train_time)
	output_string = " "
	for t in range(train_time):
		iters[t] = play_one(pmodel = pmodel, vmodel = vmodel, gamma = gamma, line = line)
		if (t%5 == 0):
			output_string = "iter : %d\n" % iter[t]
			with open(file_name, "a") as text_file:
				text_file.write(output_string)

	with open(file_name, "a") as text_file:
	text_file.write("finish!!!!!!!!!!!!!!!")	

def main():
	D = 30
	pmodel = PolicyModel(D, [50, 20], [50, 20])
	session = tf.InteractiveSession()

	pmodel.set_session(session)
	pmodel.init_vars()
	gamma = 0.99

	totaleps, pmodel = random_search(pmodel, gamma)

	print("max reward:", np.max(totaleps))
	with open(file_name, "a") as text_file:
		text_file.write("max reward:" + np.array_str(np.max(totaleps)) + "\n")
		text_file.write(np.array_str(totaleps)+ "\n\n")

if __name__ == '__main__':
	main()
