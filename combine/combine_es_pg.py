import os
import sys
import numpy as np
import tensorflow as tf
from ESwNN_cuda_128 import ESwNN_train, build_net
from pg_ESwNN import HiddenLayer, PolicyModel, ValueModel, play_one, pg_train

file_name = 'combine_output.txt'

def train_combine(save_network = False):
	if save_network:
		N_GENERATION_OF_ES = 60
	else:
		N_GENERATION_OF_ES = 1
	print("ES part")
	net_shapes, net_params, b_p, line = ESwNN_train(N_GENERATION = N_GENERATION_OF_ES)
	gamma = 0.8
	D = net_shapes[0][0]
	print("\n\n\n\n pg part")
	init = tf.global_variables_initializer()
	session = tf.InteractiveSession()
	session.run(init)
	pmodel = PolicyModel(D = D, net_params = b_p, net_shapes = net_shapes, session = session, save_network = save_network)
	vmodel = ValueModel(D = D, hidden_layer_size = [30, 10], session = session)
	pg_train(pmodel = pmodel, vmodel = vmodel, gamma = gamma, line = [51, 80], f_n = file_name)

if __name__ == '__main__':
	train_combine()
