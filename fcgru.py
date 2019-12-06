# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.compat.v1.nn.rnn_cell import RNNCell

_BIAS_G3 = "bias_gate_3"
_BIAS_G2 = "bias_gate_2"
_BIAS_G1 = "bias_gate_1"
_WEIGHTS_G1 = "kernel_gate_1"
_WEIGHTS_G1H = "kernel_gate_1_h"
_WEIGHTS_G2 = "kernel_gate_2"
_WEIGHTS_G2H = "kernel_gate_2_h"
_WEIGHTS_G3 = "kernel_gate_3"
_WEIGHTS_G3H = "kernel_gate_3_h"
_BIAS_FC0 = "bias_fc0"
_WEIGHTS_FC0 = "kernel_fc0"
_BIAS_FC1 = "bias_fc1"
_WEIGHTS_FC1 = "kernel_fc1"
_BIAS_FC2 = "bias_fc2"
_WEIGHTS_FC2 = "kernel_fc2"

def getW(name, dim1, dim2, init, dtype):
	return vs.get_variable(name, [dim1, dim2], dtype=dtype, initializer=init)
def getB(name, dim, init, dtype):
	return vs.get_variable(name, [dim], dtype=dtype, initializer=init)

class FCGRU(RNNCell):
	'''
	Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078) enriched with Fully Connected layers
	'''
	def __init__(self,
		num_units,
		fc_units,
		drop,
		activation=None,
		reuse=None,
		kernel_initializer=None,
		bias_initializer=None):
		super(FCGRU, self).__init__(_reuse=reuse)
		self._num_units = num_units
		self._fc_units = fc_units
		self._drop = drop
		self._activation = activation or math_ops.tanh
		self._kernel_initializer = kernel_initializer
		self._bias_initializer = bias_initializer

	@property
	def state_size(self):
		return self._num_units

	@property
	def output_size(self):
		return self._num_units

	def call(self, inputs, state):
		with vs.variable_scope("gates"):  
			bias_ones = self._bias_initializer
			if self._bias_initializer is None:
				dtype = [a.dtype for a in [inputs, state]][0]
				bias_ones = init_ops.zeros_initializer(dtype=dtype)
	
			b_fc1 = getB(_BIAS_FC1, self._fc_units, bias_ones, dtype)
			weights_fc1 = getW(_WEIGHTS_FC1, inputs.shape[1],  self._fc_units, self._kernel_initializer, dtype)

			b_fc2 = getB(_BIAS_FC2, int(self._fc_units*2), bias_ones, dtype)
			weights_fc2 = getW(_WEIGHTS_FC2, self._fc_units, int(self._fc_units*2) , self._kernel_initializer, dtype)
			
			b_g1 = getB(_BIAS_G1, self._num_units, bias_ones, dtype)
			weights_g1 = getW(_WEIGHTS_G1, int(self._fc_units*2), self._num_units, self._kernel_initializer, dtype)
			weights_g1h = getW(_WEIGHTS_G1H, self._num_units, self._num_units, self._kernel_initializer, dtype)

			b_g2 = getB(_BIAS_G2, self._num_units, bias_ones, dtype)
			weights_g2 = getW(_WEIGHTS_G2, int(self._fc_units*2),  self._num_units, self._kernel_initializer, dtype)
			weights_g2h = getW(_WEIGHTS_G2H, self._num_units, self._num_units, self._kernel_initializer, dtype)

			b_g3 = getB(_BIAS_G3, self._num_units, bias_ones, dtype)
			weights_g3 = getW(_WEIGHTS_G3, int(self._fc_units*2),  self._num_units, self._kernel_initializer, dtype)
			weights_g3h = getW(_WEIGHTS_G3H, self._num_units, self._num_units, self._kernel_initializer, dtype)

			# Fully Connected Layers
			fc1 = math_ops.tanh( math_ops.matmul(inputs, weights_fc1) + b_fc1 )
			fc1 = tf.nn.dropout(fc1,rate=self._drop)
			
			fc2 = math_ops.tanh( math_ops.matmul(fc1, weights_fc2) + b_fc2 )
			fc2 = tf.nn.dropout(fc2,rate=self._drop)

			# Update Gate
			zt = math_ops.sigmoid( math_ops.matmul(fc2, weights_g1) + math_ops.matmul(state, weights_g1h) + b_g1) 
			# Reset Gate
			rt = math_ops.sigmoid( math_ops.matmul(fc2, weights_g2) + math_ops.matmul(state, weights_g2h) + b_g2)	
			# Memory content 
			ht_c = self._activation( math_ops.matmul(fc2, weights_g3) + math_ops.matmul(rt * state, weights_g3h) + b_g3)
			# New hidden state
			ht = (1-zt) * state + zt * ht_c
			
			return ht, ht