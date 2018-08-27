'''
Written by Evan Archer that implements the normalizing flow
'''

import theano
import lasagne

# theano.config.optimizer = 'fast'
# theano.config.exception_verbosity='high'

import theano.tensor as T
import theano.tensor.nlinalg as Tla
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np
from numpy.random import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cPickle
import sys

from theano.tensor.slinalg import expm

from scipy import stats

#sys.path.append('../lib/')
#from loadmat import loadmat

''' TODO: Lasagne parameters must be shared variables, but we can get around that
    by simply not calling the add_param function. This is exactly what we want
    if we *don't* want to use a NN to parameterize things.
'''

class RadialFlow(lasagne.layers.Layer):
    def __init__(self, incoming, z0=lasagne.init.Normal(10), alpha=lasagne.init.Constant(1.0), beta=lasagne.init.Constant(0.54132485461291802), **kwargs):
        super(RadialFlow, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = 1
        self.z0 = self.add_param(z0, (1,num_inputs), name='z0')
        self.z0 = T.addbroadcast(self.z0,0)
        self.alpha = self.add_param(alpha, (self.num_units,), name='alpha') # scalar
        self.alpha = T.addbroadcast(self.alpha,0)
        beta0 = self.add_param(beta, (self.num_units,), name='beta')
        beta0 = T.addbroadcast(beta0,0)
        # enforce the constraint beta >= -alpha, following Rezende 2015
        self.beta = -self.alpha + T.nnet.softplus(beta0)

    def get_output_for(self, input, det=False, **kwargs):
        return input + self.beta * (self.h(input) * (input - self.z0).T).T

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def h(self, z):
        r = (z - self.z0).norm(2, axis=1)
        return 1/(self.alpha + r)

    def hprime(self, z):
        r = (z - self.z0).norm(2, axis=1)
        return  - 1 / (self.alpha + r)**2

    def det(self, input):
        '''Compute Jacobian determinant'''
        d = input.shape[1]
        r = (input - self.z0).norm(2, axis=1)
        bhp = self.beta * self.h(input)
        sep_det = (d-1)*T.log(T.abs_(1+bhp))+ T.log(T.abs_(1 + bhp + self.beta*self.hprime(input) * r ))
        return sep_det.sum()

class PlanarFlow(lasagne.layers.Layer):
    def __init__(self, incoming, w=lasagne.init.Orthogonal(1), u=lasagne.init.Constant(0.0), b=lasagne.init.Normal(2), **kwargs):
        super(PlanarFlow, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = 1
        self.w = self.add_param(w, (1,num_inputs), name='w')
        self.w = T.addbroadcast(self.w,0)

        self.u0 = self.add_param(u, (1,num_inputs), name='u0')
        self.u0 = T.addbroadcast(self.u0,0)

        # enforce the constraint on u, following Rezende 2015
        self.u = self.u0 + (-1 + T.log(1+T.exp(self.w.dot(self.u0.T))) - self.w.dot(self.u0.T))*(self.w/(self.w.norm(2)**2))

        self.b = self.add_param(b, (self.num_units,), name='b') # scalar
        self.b = T.addbroadcast(self.b,0)

    def get_output_for(self, input, **kwargs):
        return input + self.u * self.h(T.dot(input, self.w.T) + self.b)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def h(self, z):
        return T.tanh(z)

    def hprime(self,z):
        return 1 - T.tanh(z)**2

    def det(self, input):
        '''Compute Jacobian determinant'''
        psi = self.hprime(T.dot(input, self.w.T) + self.b) * self.w
        sep_det = T.log(T.abs_(1+psi.dot(self.u.T)))
        return sep_det.sum()


class OrthogonalFlow(lasagne.layers.Layer):
    def __init__(self, incoming, uvec=lasagne.init.Normal(1), b=lasagne.init.Constant(0), **kwargs):
        super(OrthogonalFlow, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        
        n = num_inputs
        n_triu_entries = (n * (n + 1)) // 2
        r = T.arange(n)
        tmp_mat = r[np.newaxis, :] + (n_triu_entries - n - (r * (r + 1)) // 2)[::-1, np.newaxis]
        triu_index_matrix = T.tril(tmp_mat.T) - r[np.newaxis,:]
        tmp_mat1 = T.tril(tmp_mat.T) - r[np.newaxis,:]
        skew_index_mat = T.tril(tmp_mat1 - T.diag(T.diag(tmp_mat1)))


        self.uvec = self.add_param(uvec, ((num_inputs-1)*(num_inputs)/2,), name='uvec')
        vec0 = T.concatenate([T.zeros(1), self.uvec])
        skw_matrix = vec0[skew_index_mat] - vec0[skew_index_mat].T

        self.U = expm(skw_matrix)

        self.b = self.add_param(b, (num_inputs,), name='b') # scalar

    def get_output_for(self, input, **kwargs):
        return self.h(T.dot(input, self.U) + self.b.dimshuffle('x', 0))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def h(self, z):
        return T.tanh(z)

    def hprime(self,z):
        return 1 - T.tanh(z)**2

    def det(self, input):
        '''Compute Jacobian determinant'''
        return T.log(T.abs_(T.nlinalg.det(self.hprime(T.dot(input, self.U) + self.b.dimshuffle('x', 0)))))


class NormalizedFlow():
    def __init__(self, dim = 2, length = 20, style = 'alternating'):
        self.input = T.matrix('X')
        self.dim = dim
        network = lasagne.layers.InputLayer((None, self.dim), self.input)
        if style == 'radial':
            for i in range(length):
                #network = RadialFlow(network, alpha=np.asarray([1.0]))
                network = RadialFlow(network)
        elif style == 'planar':
            for i in range(length):
                network = PlanarFlow(network)
        elif style == 'orthogonal':
            for i in range(length):
                network = OrthogonalFlow(network)
        elif style == 'alternating':
            st = True
            for i in range(length):
                if st:
                    st = False
                    network = PlanarFlow(network)
                else:
                    st = True
                    network = RadialFlow(network)#, alpha=np.asarray([1.0]))
        else:
            raise Exception("Unknown normalizing flow style.")        

        self.network = network

        self.thelayers = lasagne.layers.get_all_layers(self.network)

    def log_jacobian_determinant(self, input):
        zk = lasagne.layers.get_output(self.thelayers, inputs=input)
        q = T.stack([l.det(z) for l,z in zip(self.thelayers[1:], zk)])        
        return theano.scan(fn = T.sum, sequences=[q])[0].sum()

    def get_input(self):
        return self.input

    def get_output(self, input = None):
        if input is None:
            input = self.input
        return lasagne.layers.get_output(self.network, inputs=input)

    def get_params(self):
        return lasagne.layers.get_all_params(self.network)
