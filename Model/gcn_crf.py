from .basemodel import Model
from Ops.layers import *

class GCN(Model):
    def __init__(self, placeholders, layer_dim, crf_type, crf_iters, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.layer_dim = layer_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.crf_iters = crf_iters

        if crf_type == 'gaussian':
            self.crf_func = CRF_Gaussian
        elif crf_type == 'nn':
            self.crf_func = CRF_NN


        with tf.variable_scope(self.name):
            self._build()



    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.layer_dim[0],
                                            output_dim=self.layer_dim[1],
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(self.crf_func(input_dim=self.layer_dim[1],
                               output_dim=self.layer_dim[1],
                               placeholders=self.placeholders,
                               num_iters=self.crf_iters,
                               logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=self.layer_dim[1],
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))




