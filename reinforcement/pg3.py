import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning_bins import plot_running_avg
from dropout_final_load import create_NN
from laplace_get_data_new2 import read_laplace_data, read_normalized_laplace_data
from laplace_env import create_LaplaceSolver


class HiddenLayer:
	def __init__(self, M1, M2, f = tf.nn.tanh, use_bias = True):
		self.use_bias = use_bias
		with tf.name_scope('layer'):
			with tf.name_scope('weights'):
				self.W = tf.Variable(tf.random_normal(shape = (M1, M2)))
			self.use_bias = use_bias
			if use_bias:
				with tf.name_scope('biases'):
					self.b = tf.Variable(tf.random_normal(shape = (M2, )))
			self.f = f

	def forward(self, X):
		if self.use_bias:
			output = tf.matmul(X, self.W) + self.b
		else:
			output = tf.matmul(X, self.W)
		return self.f(output)



class PolicyModel:
	def __init__(self, D, K, hidden_layer_size):
		with tf.name_scope('Policy'):
			self.layers = []
			M1 = D
			for M2 in hidden_layer_size:
				layer = HiddenLayer(M1, M2)
				self.layers.append(layer)
				M1 = M2
			layer = HiddenLayer(M1, K, f = tf.nn.softmax, use_bias = False)
			self.layers.append(layer)

			with tf.name_scope('Policy_inputs'):
				self.X = tf.placeholder(tf.float32, shape = (None, D), name = 'policy_X')
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

	def set_session(self, session):
		self.session = session

	def partial_fit(self, X, actions, advantages):
		X = np.atleast_2d(X)
		actions = np.atleast_1d(actions)
		advantages = np.atleast_1d(advantages)
		self.session.run(self.train_op, feed_dict = {self.X : X, self.actions : actions, self.advantages : advantages})

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict ={self.X : X})

	def sample_action(self, X):
		p = self.predict(X)[0]
		return np.random.choice(len(p), p = p)




class ValueModel:
	def __init__(self, D, hidden_layer_size):
		with tf.name_scope('Value'):
			self.layers = []
			M1 = D
			for M2 in hidden_layer_size:
				layer = HiddenLayer(M1, M2)
				self.layers.append(layer)
				M1 = M2
			layer = HiddenLayer(M1, 1)
			self.layers.append(layer)

			with tf.name_scope('Value_inputs'):
				self.X = tf.placeholder(tf.float32, shape = (None, D), name = 'value_X')
				self.Y = tf.placeholder(tf.float32, shape = (None,), name = 'Y')

			Z = self.X
			for layer in self.layers:
				Z = layer.forward(Z)
			Y_hat = tf.reshape(Z, [-1])
			self.predict_op = Y_hat

			cost = tf.reduce_sum(tf.square(self.Y - Y_hat))

			self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

	def set_session(self, session):
		self.session = session

	def partial_fit(self, X, Y):
		X = np.atleast_2d(X)
		Y = np.atleast_1d(Y)
		self.session.run(self.train_op, feed_dict = {self.X : X, self.Y : Y})

	def predict(self, X):
		X = np.atleast_2d(X)
		return self.session.run(self.predict_op, feed_dict = {self.X : X})


def play_one_td(env, pmodel, vmodel, gamma):
	observation = env.reset()
	done = False
	totalreward = 0
	iters = 0

	while not done and iters < 200:
		action = pmodel.sample_action(observation)
		prev_observation = observation
		observation, reward, done, info = env.step(action)

		V_next = vmodel.predict(observation)
		G = reward + gamma * np.max(V_next)
		advantage = G - vmodel.predict(prev_observation)[0]
		pmodel.partial_fit(prev_observation, action, advantage)
		vmodel.partial_fit(prev_observation, G)

		if reward == 1:
			totalreward += reward
		iters += 1

	return totalreward


	'''
	error = env.reset()
	done = False
	totalreward = 0
	iters = 0

	while not done:
		action = pmodel.sample_action(error)
		prev_error = error
		error, reward, done, info = env.solve_step(action)

		V_next = vmodel.predict(error)
		G = reward + gamma * np.max(V_next)
		advantage = G - vmodel.predict(prev_observation)[0]
		pmodel.partial_fit(prev_error, action, advantage)
		vmodel.partial_fit(prev_error, G)
		iters += 1
	
	return iters

	'''

def play_one_mc(laplace_env, pmodel, vmodel, gamma, last = False):
	error = laplace_env.reset_grid()
	done = False
	totalreward = 0
	iters = 0
	states = []
	actions = []
	rewards = []
	reward = 0
	observation = laplace_env.from_error_to_input_of_NN(error, r = 0.1)
	
	all_error = []

	while not done:
		action = pmodel.sample_action(observation)
		states.append(observation)
		actions.append(action)
		rewards.append(reward - 20)

		prev_observation = observation 
		ratio = (action + 1) / 10
		
		observation, reward, done, err = laplace_env.solve_step(r = ratio)

		if reward == -5000:
			iters += 100000

		if last:
			all_error.append(err)

		iters += 1

	action = pmodel.sample_action(observation)
	states.append(observation)
	actions.append(action)
	rewards.append(reward)

	returns = []
	advantages = []
	G = 0
	for s, r in zip(reversed(states), reversed(rewards)):
		returns.append(G)
		advantages.append(G - vmodel.predict(s)[0])
		G = r + gamma * G
	returns.reverse()
	advantages.reverse()

	pmodel.partial_fit(states, actions, advantages)
	vmodel.partial_fit(states, returns)

	return iters*1000, all_error


def laplace_env_main():
	session = tf.InteractiveSession()
	print("set n!!!!!  start")
	laplace_env = create_LaplaceSolver(session, n = 500)
	D = laplace_env.trained_NN.D
	K = 10
	pmodel = PolicyModel(D, K, [])
	vmodel = ValueModel(D, [10])
	init = tf.global_variables_initializer()
	session = tf.InteractiveSession()
	session.run(init)
	pmodel.set_session(session)
	vmodel.set_session(session)

	gamma = 0.99
	N = 60
	#iters = np.empty(N)
	costs = np.empty(N)
	iters = []
	output_string = None
	output_strings = []
	for n in range(N):
		if (n == N):
			ite, all_error = play_one_mc(laplace_env, pmodel, vmodel, gamma)
		else:
			ite, _ = play_one_mc(laplace_env, pmodel, vmodel, gamma)
		iters_np = np.array(iters[max(0, n-100):(n+1)])
		#output_string = "episode:", n , "iter:", ite, "average iters:", iters[max(0, n-100):(n+1)].mean()
		output_string = "episode: %d, iter: %d, average iters: %f \n" %(n , ite, iters_np.mean())
		#output_string = "episode: %d, iter: %d " %(n , ite)
		output_strings.append(output_string)

		#iters[n] = ite
		if ite < 100000000:
			iters.append(ite)
		if len(iters)%5 == 0:
			print(output_string)
			file_name = 'pg3.text'
			with open(file_name, "a") as text_file:
				text_file.write(output_string)

	print("finish!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
	laplace_env_main()