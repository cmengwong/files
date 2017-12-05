import os
import sys
import numpy as np
import tensorflow as tf
from ESwNN_simple import ESwNN_train, build_net
from pg_ESwNN import HiddenLayer, PolicyModel

if __name__ == '__main__':
	print("ES part")
	net_shapes, net_params, b_p = ESwNN_train()

	print("\n\n\n\n\n\n\n\n\n\n\n\n pg part")
	PolicyModel(b_p, net_shapes)