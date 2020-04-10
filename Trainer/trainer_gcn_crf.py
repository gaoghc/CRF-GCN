import tensorflow as tf
from Utils.utils import *
from Ops.ops import *
from Model.gcn_crf import GCN
import time



class Trainer(object):

    def __init__(self, config):
        self.config = config
        self.weight_decay = config['weight_decay']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.dropout_prob = config['dropout_prob']
        self.early_stopping = config['early_stopping']
        self.hidden_dim = config['hidden_dim']
        self.checkpt_file = config['checkpt_file']
        self.crf_iters = config['crf_iters']
        self.crf_type = config['crf_type']

        data_dir = config['data_dir']
        dataset = config['dataset']

        self.adj, self.features, self.y_train, self.y_val, self.y_test,\
        self.train_mask, self.val_mask, self.test_mask = load_data(data_dir, dataset)
        self.features = preprocess_features(self.features)

        self.support = [preprocess_adj(self.adj)]
        num_supports = 1
        model_func = GCN


        self.placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64)),
            'labels': tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1])),
            'labels_mask': tf.placeholder(tf.int32),
            'dropout': tf.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        }

        layer_dim = [self.features[2][1]] + self.hidden_dim

        self.model = model_func(self.placeholders, layer_dim=layer_dim, crf_type=self.crf_type,
                                crf_iters=self.crf_iters, logging=True)

        self.loss, self.acc = self._build_graph()


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def _build_graph(self):
        # outputs, hidden1, hidden2 = self.model.forward(self.placeholders['features'])
        outputs = self.model.forward(self.placeholders['features'])

        loss = 0.0
        for var in self.model.layers[0].vars.values():
            loss += self.weight_decay * tf.nn.l2_loss(var)
        loss += masked_softmax_cross_entropy(outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

        accuracy = masked_accuracy(outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

        return loss, accuracy


    def train(self):

        cost_val = []
        for epoch in range(self.epochs):

            start_time = time.time()
            # Training step
            feed_dict_train = construct_feed_dict(self.features, self.support, self.y_train, self.train_mask, self.placeholders)
            feed_dict_train.update({self.placeholders['dropout']: self.dropout_prob})
            outs_train = self.sess.run([self.optimizer, self.loss, self.acc], feed_dict=feed_dict_train)

            # Validation
            feed_dict_val = construct_feed_dict(self.features, self.support, self.y_val, self.val_mask, self.placeholders)
            outs_val = self.sess.run([self.loss, self.acc], feed_dict=feed_dict_val)
            cost_val.append(outs_val[0])


            # Print results
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs_train[1]),
                  "train_acc=", "{:.5f}".format(outs_train[2]), "val_loss=", "{:.5f}".format(outs_val[0]),
                  "val_acc=", "{:.5f}".format(outs_val[1]), "time=", "{:.5f}".format(time.time() - start_time))

            if epoch > self.early_stopping and cost_val[-1] > np.mean(cost_val[-(self.early_stopping + 1):-1]):
                print("Early stopping...")
                break
        print("Optimization Finished!")

        feed_dict_test = construct_feed_dict(self.features, self.support, self.y_test, self.test_mask, self.placeholders)
        test_cost, test_acc = self.sess.run([self.loss, self.acc], feed_dict=feed_dict_test)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc))

        self.saver.save(self.sess, self.checkpt_file)

    def test(self):
        self.saver.restore(self.sess, self.checkpt_file)

        feed_dict_test = construct_feed_dict(self.features, self.support, self.y_test, self.test_mask, self.placeholders)
        test_cost, test_acc = self.sess.run([self.loss, self.acc], feed_dict=feed_dict_test)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc))


