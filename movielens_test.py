import itertools
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from smf import SMF

num_user = 943
num_item = 1682

n_trained_data = 10000000
batch_sizes = [256, 512, 1024]
# reg_lambdas = [0, 1e-10, 1e-7, 1e-5]
reg_lambdas = [0, 1e-5, 1e-3, 1e-1, 1]
learning_rates = [0.001, 0.0001]
latent_dims = [25, 50, 100]

def read_dataset():
    M = np.zeros([num_user, num_item])
    with open('./data/ml-100k/u.data', 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            user_id = int(tokens[0]) - 1  # 0 base index
            item_id = int(tokens[1]) - 1
            rating = int(tokens[2])
            M[user_id, item_id] = rating
    return M


def train_test_validation():
    M = read_dataset()

    num_rating = np.count_nonzero(M)
    idx = np.arange(num_rating)
    np.random.seed(0)
    np.random.shuffle(idx)

    train_prop = 0.9
    valid_prop = train_prop * 0.05
    train_idx = idx[:int((train_prop-valid_prop) * num_rating)]
    valid_idx = idx[int((train_prop-valid_prop) * num_rating):int(train_prop * num_rating)]
    test_idx = idx[int(0.9 * num_rating):]

    for learning_rate, batch_size, reg_lambda, latent_dim in itertools.product(learning_rates, batch_sizes, reg_lambdas, latent_dims):
        result_path = "{0}_{1}_{2}_{3}".format(learning_rate, batch_size, reg_lambda, latent_dim)
        if not os.path.exists(result_path + "/model.ckpt.index"):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            with tf.Session(config=config) as sess:
                n_steps = int(n_trained_data / batch_size)
                model = SMF(sess, M, latent_dim, learning_rate=learning_rate, batch_size=batch_size, reg_lambda=reg_lambda)
                best_rmse, best_mae = model.train_test_validation(
                    M, train_idx=train_idx, test_idx=test_idx, valid_idx=valid_idx, n_steps=n_steps, result_path=result_path)

                print("Best RMSE = {0}, best MAE = {1}".format(best_rmse, best_mae))
                with open('result.csv', 'a') as f:
                    f.write("{0},{1},{2},{3},{4},{5}\n".format(learning_rate, batch_size, reg_lambda, latent_dim, best_rmse, best_mae))
        tf.reset_default_graph()

if __name__ == '__main__':
    train_test_validation()
