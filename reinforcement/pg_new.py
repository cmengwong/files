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
	def __init__(self, M1, M2, f = tf.nn.tanh, use_bias = True, zeros = False):
		if zeros:
			W = np.zeros((M1, M2)).astype(np.float32)
			self.W = tf.Variable(W)
		else:
			self.W = tf.Variable(tf.random_normal(shape = (M1, M2)))

		self.params = [self.W]
		self.use_bias = use_bias

		if use_bias:
			self.b = tf.Variable(np.zeros(M2).astype(np.float32))
			self.params.append(self.b)

		self.f = f

	def forward(self, X):
		if self.use_bias:
			output = tf.matmul(X, self.W) + self.b
		else:
			output = tf.matmul(X, self.W)
		return self.f(output)


class PolicyModel:
	def __init__(self, D, hidden_layer_sizes_mean= [], hidden_layer_sizes_var = []):
		self.D = D
		self.hidden_layer_sizes_var = hidden_layer_sizes_var
		self.hidden_layer_sizes_mean = hidden_layer_sizes_mean

		##########initialize the mean network############ 
		self.mean_layers = []
		M1 = D
		for M2 in hidden_layer_sizes_mean:
			layer = HiddenLayer(M1, M2)
			self.mean_layers.append(layer)
			M1 = M2

		###final layer###
		layer = HiddenLayer(M1, 1, lambda x:x, use_bias = False, zeros =True)
		self.mean_layers.append(layer)


		##########initialize the var network############## 
		self.var_layers = []
		M1 = D
		for M2 in hidden_layer_sizes_var:
			layer = HiddenLayer(M1, M2)
			self.var_layers.append(layer)
			M1 = M2

		###final layer###
		layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias = False, zeros =True)
		self.var_layers.append(layer)



		###############record the network##################
		self.params = []
		for layer in (self.mean_layers + self.var_layers):
			self.params += layer.params

		###############set some tensorflow term############
		self.X = tf.placeholder(tf.float32, shape = (None, D), name = 'X')
		self.actions = tf.placeholder(tf.float32, shape = (None,), name = 'actions')
		self.advantages = tf.placeholder(tf.float32, shape = (None,), name = 'advantages')

		def get_output(layers):
			Z = self.X
			for layer in layers:
				Z = layer.forward(Z)
			return tf.reshape(Z, [-1])

		mean = get_output(self.mean_layers)
		var = get_output(self.var_layers) + 1e-4 # for smoothing

		norm = tf.contrib.distributions.Normal(mean, var)
		self.predict_op = tf.clip_by_value(norm.sample(), 0, 1)


	def set_session(self, session):
		self.session = session

	def init_vars(self):
		init_op = tf.variable_initializer(self.params)
		self.session.run(init_op)

	def predict(self, X):
		X = np.atleast_2d(X)
		X = self.tf.transform(X)
		return self.session.run(self.predict_op, feed_dict = {self.X: X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return p

	def copy(self):
		clone = PolicyModel(self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_var)
		clone.set_session(self.session)
		clone.init_vars()
		clone.copy_from(self)
		return clone

	def copy_from(self, other):
		ops = []
		my_params = self.my_params
		other_params = other.my_params
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


def play_multiple_episodes(times = 3, pmodel, gamma, print_iters=False):
	totaleps = np.emtpy(times)

	for i in range(times):
		totaleps[i] = play_one(env, pmodel, gamma)

		if print_iters:
			print(i, "avg so far:", totaleps[:(i+1)].mean())

	avg_totaleps = totaleps.means()
	print("avg total eps:", avg_totaleps)
	return avg_totaleps


def random_search(env, pmodel, gamma):
	totaleps = []
	best_avg_totaleps = float('-inf')
	best_pmodel = pmodel
	num_episodes_per_param_test = 3
	for t in range(10):
		tmp_pmodel = best_pmodel.copy()
		tmp_pmodel.perturb_params()
		avg_totaleps = play_multiple_episodes(num_episodes_per_param_test, tmp_pmodel, gamma)
		totaleps.append(avg_totaleps)

		if(avg_totaleps > best_avg_totaleps)
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

if __name__ == '__main__':
	main()
