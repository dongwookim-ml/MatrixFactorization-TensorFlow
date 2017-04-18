import tensorflow as tf
import numpy as np

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)


class SMF:
    def __init__(self, sess, M, k, learning_rate=0.001, batch_size=256, reg_lambda=0.01):
        self.sess = sess
        self.n, self.m = M.shape
        self.k = k
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        self.build_graph()

    def build_graph(self):
        self.u_idx = tf.placeholder(tf.int32, [None])
        self.v_idx = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.float32, [None])

        self.U = weight_variable([self.n, self.k], 'U')
        self.V = weight_variable([self.m, self.k], 'V')
        self.U_bias = weight_variable([self.n], 'U_bias')
        self.V_bias = weight_variable([self.m], 'V_bias')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)
        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed)

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.l2_loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.l2_loss, self.reg)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        self.train_step_u = self.optimizer.minimize(self.reg_loss, var_list=[self.U, self.U_bias])
        self.train_step_v = self.optimizer.minimize(self.reg_loss, var_list=[self.V, self.V_bias])

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("L2-Loss", self.l2_loss)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

    def construct_feeddict(self, u_idx, v_idx, M):
        return {self.u_idx:u_idx, self.v_idx:v_idx, self.r:M[u_idx, v_idx]}

    def train_test_validation(self, M, train_idx, test_idx, valid_idx, n_steps=100000, result_path='result/'):
        nonzero_u_idx = M.nonzero()[0]
        nonzero_v_idx = M.nonzero()[1]

        train_size = train_idx.size
        trainM = np.zeros(M.shape)
        trainM[nonzero_u_idx[train_idx], nonzero_v_idx[train_idx]] = M[
            nonzero_u_idx[train_idx], nonzero_v_idx[train_idx]]

        best_val_rmse = np.inf
        best_val_mae = np.inf
        best_test_rmse = 0
        best_test_mae = 0

        train_writer = tf.summary.FileWriter(result_path + '/train', graph=self.sess.graph)
        valid_writer = tf.summary.FileWriter(result_path + '/validation', graph=self.sess.graph)
        test_writer = tf.summary.FileWriter(result_path + '/test', graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        for step in range(1, n_steps):
            batch_idx = np.random.randint(train_size, size=self.batch_size)
            u_idx = nonzero_u_idx[train_idx[batch_idx]]
            v_idx = nonzero_v_idx[train_idx[batch_idx]]
            feed_dict = self.construct_feeddict(u_idx, v_idx, trainM)

            self.sess.run(self.train_step_v, feed_dict=feed_dict)
            _, rmse, mae, summary_str = self.sess.run(
                [self.train_step_u, self.RMSE, self.MAE, self.summary_op], feed_dict=feed_dict)
            train_writer.add_summary(summary_str, step)

            if step % int(n_steps / 100) == 0:
                valid_u_idx = nonzero_u_idx[valid_idx]
                valid_v_idx = nonzero_v_idx[valid_idx]
                feed_dict = self.construct_feeddict(valid_u_idx, valid_v_idx, M)
                rmse_valid, mae_valid, summary_str = self.sess.run(
                    [self.RMSE, self.MAE, self.summary_op], feed_dict=feed_dict)

                valid_writer.add_summary(summary_str, step)

                test_u_idx = nonzero_u_idx[test_idx]
                test_v_idx = nonzero_v_idx[test_idx]
                feed_dict = self.construct_feeddict(test_u_idx, test_v_idx, M)
                rmse_test, mae_test, summary_str = self.sess.run(
                    [self.RMSE, self.MAE, self.summary_op], feed_dict=feed_dict)

                test_writer.add_summary(summary_str, step)

                print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                    step, rmse, mae))
                print("         | Valid  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_valid, mae_valid))
                print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_test, mae_test))

                if best_val_rmse > rmse_valid:
                    best_val_rmse = rmse_valid
                    best_test_rmse = rmse_test

                if best_val_mae > mae_valid:
                    best_val_mae = mae_valid
                    best_test_mae = mae_test

        self.saver.save(self.sess, result_path + "/model.ckpt")
        return best_test_rmse, best_test_mae
