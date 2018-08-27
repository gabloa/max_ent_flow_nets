import lasagne
import theano
import theano.tensor as T
import theano.tensor.fft # for circulant matrix
import numpy as np

'''
custom lasagne layers to map data to manifold
Yuanjun Gao
08/25/2016 
'''

# circulant matrix multiplication                                                                                                                                
class CircMatLayer(lasagne.layers.Layer):
    def __init__(self, incoming, C=lasagne.init.Normal(0.01), **kwargs):
        super(CircMatLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.C = self.add_param(C, (1,num_inputs), name='C')
        self.C = T.addbroadcast(self.C, 0)

    def get_output_for(self, input, **kwargs):
        C_fft = T.addbroadcast(theano.tensor.fft.rfft(self.C), 0)
        z_fft = theano.tensor.fft.rfft(input)
        Cz_fft = z_fft
        Cz_fft = T.set_subtensor(Cz_fft[:,:,0], z_fft[:,:,0] * C_fft[:,:,0] - z_fft[:,:,1] * C_fft[:,:,1])
        Cz_fft = T.set_subtensor(Cz_fft[:,:,1], z_fft[:,:,0] * C_fft[:,:,1] + z_fft[:,:,1] * C_fft[:,:,0])
        rlt = theano.tensor.fft.irfft(Cz_fft)
        return rlt

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def det(self, input):
        # to avoid complication, right now only accept even number of nodes!!!
        C_fft =T.addbroadcast(theano.tensor.fft.rfft(self.C), 0)
        C_fft_norm = T.log((C_fft ** 2).sum(2)) / 2.0
        C_fft_logdet = C_fft_norm[0,0] + C_fft_norm[0,-1] + C_fft_norm[0,1:-1].sum() * 2
        return input.shape[0] * C_fft_logdet



class CircMatLayerSparse2D(lasagne.layers.Layer):
    def __init__(self, incoming, kernel_shape=[5,5], input_shape=[50,50], C=lasagne.init.Normal(0.01), **kwargs):
        super(CircMatLayerSparse2D, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.kernel_shape = kernel_shape
        self.input_shape = input_shape
        self.C = self.add_param(C, (1,kernel_shape[0] * kernel_shape[1]), name='C')
        self.C = T.addbroadcast(self.C, 0)
        #self.C_pad = self.C.reshape(kernel_shape, ndim=2)
        self.C_pad = T.zeros(input_shape)
        self.C_pad = T.set_subtensor(self.C_pad[:kernel_shape[0], :kernel_shape[1]], self.C.reshape(kernel_shape,ndim=2))
        self.C_pad = self.C_pad.reshape([1, input_shape[0] * input_shape[1]])

    def get_inverse_for(self, input, **kwargs):
        C_fft = T.addbroadcast(theano.tensor.fft.rfft(self.C_pad), 0)
        C_fft_norm2 = C_fft[:,:,0] ** 2 + C_fft[:,:,1] ** 2
        C_fft_inv = C_fft
        C_fft_inv = T.set_subtensor(C_fft_inv[:,:,0], C_fft[:,:,0] / C_fft_norm2)
        C_fft_inv = T.set_subtensor(C_fft_inv[:,:,1], -C_fft[:,:,1] / C_fft_norm2)
        z_fft = theano.tensor.fft.rfft(input)
        Cz_fft = z_fft
        Cz_fft = T.set_subtensor(Cz_fft[:,:,0], z_fft[:,:,0] * C_fft_inv[:,:,0] - z_fft[:,:,1] * C_fft_inv[:,:,1])
        Cz_fft = T.set_subtensor(Cz_fft[:,:,1], z_fft[:,:,0] * C_fft_inv[:,:,1] + z_fft[:,:,1] * C_fft_inv[:,:,0])
        rlt = theano.tensor.fft.irfft(Cz_fft)
        return rlt


    def get_output_for(self, input, **kwargs):
        C_fft = T.addbroadcast(theano.tensor.fft.rfft(self.C_pad), 0)
        z_fft = theano.tensor.fft.rfft(input)
        Cz_fft = z_fft
        Cz_fft = T.set_subtensor(Cz_fft[:,:,0], z_fft[:,:,0] * C_fft[:,:,0] - z_fft[:,:,1] * C_fft[:,:,1])
        Cz_fft = T.set_subtensor(Cz_fft[:,:,1], z_fft[:,:,0] * C_fft[:,:,1] + z_fft[:,:,1] * C_fft[:,:,0])
        rlt = theano.tensor.fft.irfft(Cz_fft)
        return rlt

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def det(self, input):
        # to avoid complication, right now only accept even number of nodes!!!
        C_fft =T.addbroadcast(theano.tensor.fft.rfft(self.C_pad), 0)
        C_fft_norm = T.log((C_fft ** 2).sum(2)) / 2.0
        C_fft_logdet = C_fft_norm[0,0] + C_fft_norm[0,-1] + C_fft_norm[0,1:-1].sum() * 2
        return input.shape[0] * C_fft_logdet




class PlanarEleLayer(lasagne.layers.Layer):
    def __init__(self, incoming, w=lasagne.init.Orthogonal(1), u=lasagne.init.Constant(0.0), b=lasagne.init.Normal(2), **kwargs):
        super(PlanarEleLayer, self).__init__(incoming, **kwargs)
        self.w = self.add_param(w, (1,1), name='w')
        self.w = T.addbroadcast(T.addbroadcast(self.w, 0),1)
        self.u0 = self.add_param(u, (1,1), name='u0')
        self.u = self.u0 + (-1 + T.log(1+T.exp(self.w * self.u0)) - self.w * self.u0) / self.w
        self.u = T.addbroadcast(T.addbroadcast(self.u, 0),1)
        self.b = self.add_param(b, (1,1), name='b')
        self.b = T.addbroadcast(T.addbroadcast(self.b, 0),1)

    def get_output_for(self, input, **kwargs):
        return input + self.u * self.h(input * self.w + self.b)

    def h(self, z):
        return T.tanh(z)

    def hprime(self, z):
        return 1 - T.tanh(z)**2

    def det(self, input):
        psi = self.hprime(input * self.w + self.b) * self.w
        sep_det = T.log(T.abs_(1 + psi * self.u))
        return sep_det.sum()

# Linear transformation layer with only diagonal transformation
class LinDiagLayer(lasagne.layers.Layer):
    def __init__(self, incoming, W=lasagne.init.Normal(0.01), b=lasagne.init.Normal(0.01), **kwargs):
        super(LinDiagLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[1]
        self.W = self.add_param(W, (1,num_inputs), name='W')
        self.W = T.addbroadcast(self.W, 0)
        self.b = self.add_param(b, (1,num_inputs), name='b')
        self.b = T.addbroadcast(self.b, 0)

    def get_output_for(self, input, **kwargs):
        return input * T.exp(self.W) + self.b

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def det(self, input):
        return input.shape[0] * self.W.sum()

# Exponential transformation layer used to enforce positivity constraint
class ExpLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.exp(input)

    def det(self, input):
        '''LOG determinant'''
        return input.sum().sum()

class ReluEleLayer(lasagne.layers.Layer):
    def __init__(self, incoming, x0=lasagne.init.Normal(0.01), a0=lasagne.init.Normal(0.01), a1 = lasagne.init.Normal(0.01), **kwargs):
        super(ReluEleLayer, self).__init__(incoming, **kwargs)
        self.x0 = self.add_param(x0, (1,1), name='x0')
        self.x0 = T.addbroadcast(T.addbroadcast(self.x0, 0),1)
        self.a0 = self.add_param(a0, (1,1), name='a0')
        self.a0 = T.addbroadcast(T.addbroadcast(self.a0, 0),1)
        self.a1 = self.add_param(a1, (1,1), name='a1')
        self.a1 = T.addbroadcast(T.addbroadcast(self.a1, 0),1)
        self.b0 = T.exp(self.a0)
        self.b1 = T.exp(self.a1)

    def get_output_for(self, input, **kwargs):
        return self.b1 * T.nnet.relu(input - self.x0, self.b0) + self.x0

    def get_inverse_for(self, input, **kwargs):
        return 1/self.b1 * T.nnet.relu(input - self.x0, 1 / self.b0) + self.x0

    def det(self, input):
        return ((input < self.x0) * self.a0 + self.a1).sum()


# Linear transformation layer
class LinEleLayer(lasagne.layers.Layer):
    def __init__(self, incoming, w=lasagne.init.Normal(0.01, 1.0), b=lasagne.init.Normal(0.01), **kwargs):
        super(LinEleLayer, self).__init__(incoming, **kwargs)
        self.w = self.add_param(w, (1,1), name='w')
        self.w = T.addbroadcast(T.addbroadcast(self.w, 0), 1)
        self.b = self.add_param(b, (1,1), name='b')
        self.b = T.addbroadcast(T.addbroadcast(self.b, 0), 1)

    def get_output_for(self, input, **kwargs):
        return self.w * input + self.b 

    def get_inverse_for(self, input):
        return (input - self.b) / self.w
    
    def det(self, input):
        return input.shape[0] * input.shape[1] * T.log(T.abs_(self.w))

# Tanh transformation
class TanhLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.tanh(input)

    def get_inverse_for(self, input):
        return T.arctanh(input)

    def det(self, input):
        return T.log(1 - T.tanh(input) ** 2).sum()

    


# Normalization transformation layer
class NormalizeLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return input / (T.sqrt((input ** 2).sum(axis=1)).reshape((input.shape[0], 1)))


class SimplexLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.exp(input) / T.exp(input).sum(axis=1).reshape([input.shape[0], 1])


class SimplexBijectionLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        ex = T.exp(input)
        den = ex.sum(axis=1) + 1.0
        return T.concatenate([ex / den.reshape([den.shape[0], 1]), 1.0 / den.reshape([den.shape[0], 1])], axis=1)

    def det(self, input):
        '''LOG determinant'''
        ex = T.exp(input)
        den = ex.sum(axis=1) + 1.0
        log_dets = T.log(1.0 - ex.sum(axis=1) / den) - input.shape[1]*T.log(den)+input.sum(axis=1) #- T.log(input.shape[1] + 1.0) / 2.0
        return log_dets.sum()

class CholeskyLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        n = 3 #wishart is of size nxn, this should be automated in the future
        batch_size = T.shape(input)[0]
        a = T.zeros((batch_size, n, n))
        batch_idx = T.extra_ops.repeat(T.arange(batch_size), n)
        diag_idx = T.tile(T.arange(n), batch_size)
        b = T.set_subtensor(a[batch_idx, diag_idx, diag_idx], T.flatten(T.exp(input[:, :n]))) #diagonal elements
        cols = np.concatenate([np.array(range(i), dtype=np.uint) for i in xrange(n)])
        rows = np.concatenate([np.array([i] * i, dtype=np.uint) for i in xrange(n)])
        cols_idx = T.tile(T.as_tensor_variable(cols), batch_size)
        rows_idx = T.tile(T.as_tensor_variable(rows), batch_size)
        batch_idx = T.extra_ops.repeat(T.arange(batch_size), len(cols))
        chol_L = T.set_subtensor(b[batch_idx, rows_idx, cols_idx], T.flatten(input[:, n:]))
        return chol_L

    def det(self, input):
        '''LOG determinant'''
        n = 3  # wishart is of size nxn, this should be automated in the future
        batch_size = T.shape(input)[0]
        a = T.zeros((batch_size, n, n))
        batch_idx = T.extra_ops.repeat(T.arange(batch_size), n)
        diag_idx = T.tile(T.arange(n), batch_size)
        logdets = T.set_subtensor(a[batch_idx, diag_idx, diag_idx], T.flatten(input[:, :n])).sum(axis=[1, 2])
        return logdets.sum()

    def get_output_shape_for(self, input_shape):
        #n = 3  # wishart is of size nxn, this should be automated in the future
        d = input_shape[1]
        n = int(-0.5 + np.sqrt(0.25 + 2 * d) + 0.1)# 0.1 is added since the int function always rounds down
        return [input_shape[0], n, n]

class CholeskyProdLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        Sigmas = T.batched_dot(input, input.dimshuffle(0, 2, 1))
        return Sigmas

    def det(self, input):
        '''LOG determinant'''
        #logdets, _ = theano.scan(
            #fn=lambda mat: mat.shape[0] * T.log(2) + T.sum((T.arange(mat.shape[0]) + 1.0) * T.log(T.diag(mat))),
            #outputs_info=None, sequences=input)
        n = 3 # wishart is of size nxn, this should be automated in the future
        batch_size = T.shape(input)[0]
        cols = np.concatenate([np.array(range(i), dtype=np.uint) for i in xrange(n)])
        rows = np.concatenate([np.array([i] * i, dtype=np.uint) for i in xrange(n)])
        cols_idx = T.tile(T.as_tensor_variable(cols), batch_size)
        rows_idx = T.tile(T.as_tensor_variable(rows), batch_size)
        batch_idx = T.extra_ops.repeat(T.arange(batch_size), len(cols))
        chol_L_diags = T.set_subtensor(input[batch_idx, rows_idx, cols_idx], 0).sum(axis=1)
        logdets = input.shape[1] * T.log(2) + T.sum((T.arange(input.shape[1]) + 1.0) * T.log(chol_L_diags), axis=1)
        return logdets.sum()

# Softplus transformation layer used to enforce positivity constraint
class SoftplusLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return T.nnet.softplus(input) #T.log(1 + T.exp(input))

    def det(self, input):
        '''LOG determinant'''
        return -T.nnet.softplus(-input).sum().sum() # (- T.log(1 + T.exp(-input))).sum().sum()
