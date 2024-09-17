import tensorflow as tf
import numpy as np
#from tensorflow.contrib import layers
import scipy.io as sio
from sklearn.preprocessing import minmax_scale


def next_batch(data_, _index_in_epoch, batch_size, num_views, _epoch_completed):
    # this function is used to get data in next_batch
    # the number of views is started with 0
    _num_examples = data_['0'].shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size

    if _index_in_epoch > _num_examples:
        # finish current epoch
        _epoch_completed += 1 
        # shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        for i in range(0, num_views):
            data_[str(i)] = data_[str(i)][perm]
        
        # start new epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    data = {}
    for i in range(0, num_views):
        data[str(i)] = data_[str(i)][start:end]
    return data, _index_in_epoch, _epoch_completed

class AE_full(object):
    def __init__(self, enc_dim_list, dec_dim_list, num_views=6 ,learning_rate=1e-3, batch_size=210, reg=None,
                 model_path=None, restore_path=None, logs_path='./Net_Models_logs/MSRCV1_Pre_logs_multiview'):

        self.enc_dim_list = enc_dim_list # matrix
        self.dec_dim_list = dec_dim_list # matrix
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.batch_size = batch_size
        self.iter = 0
        self.num_views = num_views
        weights = self._initialize_weights()

        self.x = {}

        for i in range(0, self.num_views):
            modality = str(i)
            self.x[modality] = tf.placeholder(tf.float32, [None, enc_dim_list[i][0]])

        latents = self.encoder(self.x, weights, num_views)

        self.x_r = self.decoder(latents, weights, num_views)
        self.saver = tf.train.Saver()

        # loss
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.x_r['0'], self.x['0']), 2.0))
        for i in range(1, num_views):
            modality = str(i)
            self.cost = self.cost + 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.x_r[modality], self.x[modality]), 2.0))
        tf.summary.scalar("l2_loss", self.cost)

        self.merged_summary_op = tf.summary.merge_all()

        self.loss = self.cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)                     
        init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        self.summary_weiter = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())

        t_vars = tf.trainable_variables()
        for var in t_vars:
            print(var.name)
            print(var.shape)

    def _initialize_weights(self):
        all_weights = dict()
        
        for i in range(0, self.num_views):
            modality = str(i)
            with tf.variable_scope(modality):
                # print(modality)
                # encoder layer 1
                all_weights[modality+'_enc_w0'] = tf.get_variable(modality+"_enc_w0", shape=[self.enc_dim_list[i][0], self.enc_dim_list[i][1]],
                    initializer=tf.keras.initializers.glorot_normal(),regularizer=self.reg)
                all_weights[modality+'_enc_b0'] = tf.Variable(tf.zeros([self.enc_dim_list[i][1]], dtype=tf.float32))
                # encoder layer 2
                all_weights[modality+'_enc_w1'] = tf.get_variable(modality+"_enc_w1", shape=[self.enc_dim_list[i][1], self.enc_dim_list[i][2]],
                    initializer=tf.keras.initializers.glorot_normal(),regularizer=self.reg)
                all_weights[modality+'_enc_b1'] = tf.Variable(tf.zeros([self.enc_dim_list[i][2]], dtype=tf.float32))

                # decoder layer 1
                all_weights[modality+'_dec_w0'] = tf.get_variable(modality+"_dec_w0", shape=[self.enc_dim_list[i][2], self.dec_dim_list[i][0]],
                    initializer=tf.keras.initializers.glorot_normal(),regularizer=self.reg)
                all_weights[modality+'_dec_b0'] = tf.Variable(tf.zeros([self.dec_dim_list[i][0]], dtype=tf.float32))
                # decoder layer 2
                all_weights[modality+'_dec_w1'] = tf.get_variable(modality+"_dec_w1", shape=[self.dec_dim_list[i][0], self.dec_dim_list[i][1]],
                    initializer=tf.keras.initializers.glorot_normal(),regularizer=self.reg)
                all_weights[modality+'_dec_b1'] = tf.Variable(tf.zeros([self.dec_dim_list[i][1]], dtype=tf.float32))

        return all_weights

    def encoder(self, x, weights, num_views):
        # layer 1
        latents = {}
        for i in range(0, num_views):
            modality = str(i)
            layers1 = tf.add(tf.matmul(x[modality], weights[modality+'_enc_w0']), weights[modality+'_enc_b0'])
            layers1 = tf.nn.relu(layers1)
            # layer 2
            layers2 = tf.add(tf.matmul(layers1, weights[modality+'_enc_w1']), weights[modality+'_enc_b1'])
            layers2 = tf.nn.relu(layers2)
            latents[modality] = layers2

        return latents

    def decoder(self, z, weights, num_views):
        recons = {}
        for i in range(0, num_views):
            modality = str(i)    
            # layer 1
            layers1 = tf.add(tf.matmul(z[modality], weights[modality+'_dec_w0']), weights[modality+'_dec_b0'])
            layers1 = tf.nn.relu(layers1)
            # layer 2
            layers2 = tf.add(tf.matmul(layers1, weights[modality+'_dec_w1']), weights[modality+'_dec_b1'])
            layers2 = tf.nn.relu(layers2)
            recons[modality] = layers2

        return recons

    def partial_fit(self, X):
        feed_dict = {}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        cost, summary, _ = self.sess.run((self.cost, self.merged_summary_op, self.optimizer), feed_dict = feed_dict)
        self.summary_weiter.add_summary(summary, self.iter)
        self.iter = self.iter + 1

        return cost

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)


def pre_train(x, AE, batch_size, num_views):
    it = 0
    display_step = 50
    save_step = 20000
    _index_in_epoch = 0
    _epochs = 0

    while True:
        batch_x, _index_in_epoch, _epochs = next_batch(x,_index_in_epoch, batch_size, num_views ,_epochs)

        cost = AE.partial_fit(batch_x)
        it = it + 1

        avg_cost = cost/batch_size
        if it % display_step == 0:
            print ("epoch: %.1d" % _epochs)
            print ("cost: %.8f" % avg_cost)

        if it % save_step == 0:
            AE.save_model()
            break


def load_data(file_name):
    dataset = sio.loadmat(file_name)
    x1, x2, x3, x4, x5, x6, gt = dataset['x1'], dataset['x2'], dataset['x3'], dataset['x4'], dataset['x5'], dataset['x6'], dataset['gt']
    gt = gt.flatten()
        
    return x1, x2, x3, x4, x5, x6, gt


if __name__ == '__main__':

    x0, x1, x2, x3, x4, x5, gt = load_data('./datasets/MSRCV1.mat')
    x0 = minmax_scale(x0)
    x1 = minmax_scale(x1)
    x2 = minmax_scale(x2)
    x3 = minmax_scale(x3)
    x4 = minmax_scale(x4)
    x5 = minmax_scale(x5)
    x = {}
    x['0'] = x0
    x['1'] = x1
    x['2'] = x2
    x['3'] = x3
    x['4'] = x4
    x['5'] = x5
    num_views = 6
    enc_dim_list = [[1302, 512, 128], [48, 64, 128], [512, 256, 128], [100, 128, 128], [256, 128, 128], [210, 128, 128]]
    dec_dim_list = [[512, 1302], [64, 48], [256, 512], [128, 100], [128, 256], [128, 210]]
    batch_size = 210
    model_path = './Net_Models_logs/Model_multiview/MSRCV1_model.ckpt'

    AE = AE_full(enc_dim_list=enc_dim_list, dec_dim_list=dec_dim_list, num_views=6, learning_rate=1e-3, batch_size=batch_size, model_path=model_path)

    pre_train(x, AE, batch_size, num_views)

    


            







