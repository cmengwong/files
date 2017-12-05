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

############initial setting#############
n = 128
ep_max_step = 10000
file_name = "pg_random.txt"
_, _, normalize_data = read_normalized_laplace_data()


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
	def __init__(self, D, net_params, net_shapes):
		self.D = net_shapes[0][1]
		self.net_shapes = net_shapes
		self.net_params = net_params
		self.layers = []

		########### copy the data from the ESwNN ###############

		start = 0
		for i, shape in enumerate(net_shapes):
			n_w, n_b = shape[0] * shape[1], shape[1]
			W_E, b_E = params[start: start + n_w].reshape(shape),
					params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))
			layer = HiddenLayer(copy_from_ESwNN = True, W_E = W_E, b_E = b_E)
			self.layers.append(layer)
			start += n_w + n_b

		############### record the network ##################
		self.params = []
		for layer in (self.layers):
			self.params += layer.params

		############### set some tensorflow term ############
		self.X = tf.placeholder(tf.float32, shape = (None, D), name = 'X')
		self.actions = tf.placeholder(tf.float32, shape = (None,), name = 'actions')
		self.advantages = tf.placeholder(tf.float32, shape = (None,), name = 'advantages')

		Z = self.X
		for layer in self.layers:
			Z = layer.forward(Z)
		p_a_given_s = Z

		self.predict_op = p_a_given_s

		selected_probs = tf.log(tf.reduce_sum(p_a_given_s * tf.one_hot(self.actions, K), reduction_indices = [1]))

		cost = -tf.reduce_sum(self.advantages * selected_probs)

		self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)


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

	def copy(self):
		clone = PolicyModel(self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var)
		clone.set_session(self.session)
		clone.init_vars()
		clone.copy_from(self)
		return clone

	def copy_from(self, other):
		ops = []
		my_params = self.params
		other_params = other.params
		for p, q in zip(my_params, other_params):
			actual = self.session.run(q)
			op = p.assign(actual)
			ops.append(op)

		self.session.run(ops)

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

	def copy_network_from_ESwNN(self):



def play_one(pmodel, gamma):

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

	############## run by the network ######################### 
	for step in range(ep_max_step):
		action = pmodel.sample_action(observation)
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