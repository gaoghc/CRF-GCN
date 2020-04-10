import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        # self.vars = {}
        # self.placeholders = {}

        self.layers = []
        # self.activations = []
        #
        # self.inputs = None
        # self.outputs = None

        # self.loss = 0
        # self.accuracy = 0
        # self.optimizer = None
        # self.opt_op = None

    #     """ Wrapper for _build() """
    #     with tf.variable_scope(self.name):
    #         self._build()
    #
    # def _build(self):
    #     # raise NotImplementedError
    #     pass


    def forward(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    # def forward(self, x):
    #     hidden = x
    #     hidden_1 = x
    #     for idx, layer in enumerate(self.layers):
    #         hidden = layer(hidden)
    #         if idx == 0:
    #             hidden_1 = hidden
    #     return hidden, hidden_1

    # def forward(self, x):
    #     hidden = x
    #     hidden_1 = x
    #     hidden_2 = x
    #     for idx, layer in enumerate(self.layers):
    #         hidden = layer(hidden)
    #         if idx == 0:
    #             hidden_1 = hidden
    #         if idx == 1:
    #             hidden_2 = hidden
    #     return hidden, hidden_1, hidden_2


    # def _loss(self):
    #     raise NotImplementedError
    #
    # def _accuracy(self):
    #     raise NotImplementedError

    # def save(self, sess=None):
    #     if not sess:
    #         raise AttributeError("TensorFlow session not provided.")
    #     saver = tf.train.Saver(self.vars)
    #     save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
    #     print("Model saved in file: %s" % save_path)
    #
    # def load(self, sess=None):
    #     if not sess:
    #         raise AttributeError("TensorFlow session not provided.")
    #     saver = tf.train.Saver(self.vars)
    #     save_path = "tmp/%s.ckpt" % self.name
    #     saver.restore(sess, save_path)
    #     print("Model restored from file: %s" % save_path)
