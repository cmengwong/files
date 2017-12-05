import os
import sys
import numpy as np
import tensorflow as tf
from ESwNN_simple import ESwNN_train, build_net
from pg_EswNN import HiddenLayer, PolicyModel

if __name__ == '__main__':
	net_shapes, net_params, _ = ESwNN_train()
	PolicyModel(net_params, net_shapes)