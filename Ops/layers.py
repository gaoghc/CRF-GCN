import tensorflow as tf
from Ops.ops import *


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def ele_mul(x, y, sparse=False):

    if sparse:
        res = x.__mul__(y)
    else:
        res = tf.mul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, residual=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.residual = residual

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        # return self.act(output)
        output = self.act(output)

        if self.residual:
            output += inputs

        return output


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        output = self.act(output)


        return output


class CRF_Gaussian(Layer):

    def __init__(self, input_dim, output_dim, placeholders, num_iters, **kwargs):
        super(CRF_Gaussian, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support = placeholders['support']
        self.num_iters = num_iters

        with tf.variable_scope(self.name + '_vars'):
            self.vars['alpha'] = zeros([1], name='apha')
            self.vars['beta'] = zeros([1], name='beta')
            self.vars['sigma'] = ones([1], name='sigma')

    def _call(self, inputs):
        normalize_inputs = tf.nn.l2_normalize(inputs, axis=1)
        similarity = tf.matmul(normalize_inputs, normalize_inputs, transpose_b=True)
        similarity = similarity - tf.diag(tf.diag_part(similarity))

        sigma = tf.exp(self.vars['sigma'])
        sigma_sq = 2*sigma*sigma
        similarity = tf.exp(similarity/(2*sigma_sq))

        support = ele_mul(self.support[0], similarity, sparse=True)
        normalize = tf.sparse_reduce_sum(support, 1)
        normalize = tf.tile(tf.expand_dims(normalize, -1), [1, self.input_dim])

        alpha = tf.exp(self.vars['alpha'])
        beta = tf.exp(self.vars['beta'])


        output = inputs
        iters = tf.constant(0)
        condition = lambda iters, _ : tf.less(iters, self.num_iters)

        def body(iters, output):
            output = (inputs * beta + (dot(support, output, sparse=True) + output) * alpha) / (beta + normalize * alpha + alpha)
            return [tf.add(iters, 1), output]

        _, result = tf.while_loop(condition, body, [iters, output])


        return result


class CRF_NN(Layer):

    def __init__(self, input_dim, output_dim, placeholders, num_iters, **kwargs):
        super(CRF_NN, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support = placeholders['support']
        self.num_iters = num_iters

        with tf.variable_scope(self.name + '_vars'):
            self.vars['alpha'] = zeros([1], name='apha')
            self.vars['beta'] = zeros([1], name='beta')
            self.vars['emb_1'] = glorot([input_dim, output_dim], name='emb_1')
            self.vars['emb_2'] = glorot([input_dim, output_dim], name='emb_2')

    def _call(self, inputs):
        inputs_1 = dot(inputs, self.vars['emb_1'], sparse=False)
        inputs_2 = dot(inputs, self.vars['emb_2'], sparse=False)

        normalize_inputs_1 = inputs_1
        normalize_inputs_2 = inputs_2


        logits = tf.matmul(normalize_inputs_1, normalize_inputs_2, transpose_b=True)
        similarity = tf.nn.softmax(tf.nn.leaky_relu(logits))

        support = ele_mul(self.support[0], similarity, sparse=True)
        normalize = tf.sparse_reduce_sum(support, 1)
        normalize = tf.tile(tf.expand_dims(normalize, -1), [1, self.input_dim])

        alpha = tf.exp(self.vars['alpha'])
        beta = tf.exp(self.vars['beta'])



        output = inputs
        iters = tf.constant(0)
        condition = lambda iters, _: tf.less(iters, self.num_iters)

        def body(iters, output):
            output = (inputs * beta + (dot(support, output, sparse=True) + output) * alpha) / (
            beta + normalize * alpha + alpha)
            return [tf.add(iters, 1), output]

        _, result = tf.while_loop(condition, body, [iters, output])


        return result


