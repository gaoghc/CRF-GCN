from Trainer.trainer_gcn_crf import Trainer
from Utils import gpu_info
import os
import numpy as np
import tensorflow as tf
import argparse

# Set random seed
seed = 20
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__=='__main__':
    gpus_to_use, free_memory = gpu_info.get_free_gpu()
    print(gpus_to_use, free_memory)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use


    parser = argparse.ArgumentParser()
    parser.add_argument('--crf_iters', type=int, default='2')
    parser.add_argument('--crf_type', type=str, default='gaussian') # nn
    args = parser.parse_args()

    config = {'dataset': 'pubmed',
              'data_dir': './data',
              'crf_type': args.crf_type,
              'learning_rate': 0.01,
              'epochs': 200,
              'weight_decay': 5e-4,
              'early_stopping': 10,
              'dropout_prob': 0.5,
              'hidden_dim': [16],
              'crf_iters': args.crf_iters,
              'checkpt_file': './mod_pubmed_gcn_crf_{}.ckpt'.format(args.crf_type)}

    trainer = Trainer(config)
    trainer.train()
    trainer.test()
