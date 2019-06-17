from models.model import Model
from corpus import Corpus
from samplings import Uniform
import tensorflow as tf
import time
import numpy as np

class BPR(Model):
    def __init__(self, session, corpus, sampler, k, factor_reg, bias_reg):
        self.sampler = sampler # 选择sample方式
        self.lfactor_reg = factor_reg # 设置正则率
        self.bias_reg = bias_reg # 设置bias正则率
        self.K = k # k是latent factor model的维度

        #
        self.u, self.i, self.j, self.mf_auc, self.bprloss, self.train_op = BPR.bpr_mf(corpus.user_count, corpus.item_count,
                                                                                      k, regulation_rate = factor_reg, bias_reg = bias_reg)
        
        Model.__init__(self, corpus, session)
        print("bpr's restore...")
        self.restore()
        print("bpr - k=%d, reg_lf: %.2f, reg_bias=%.2f"%(k, factor_reg, bias_reg))

    def train(self, max_iterations, batch_size, batch_count):
        print("max_iterations: %d, batch_size: %d, batch_count: %d"%(max_iterations, batch_size, batch_count))
        corpus = self.corpus
        user_count = self.corpus.user_count
        item_count = self.corpus.item_count
        user_items = self.corpus.user_items
        item_dist = self.corpus.item_dist

        val_ratings = self.val_ratings
        test_ratings = self.test_ratings

        u, i, j, mf_auc, bprloss, train_op = self.u, self.i, self.j, self.mf_auc, self.bprloss, self.train_op

        for epoch in range(1, max_iterations+1):
            epoch_start_time = time.time()
            train_loss_vals = []

            for batch in self.sampler.generate_train_batch(user_items, val_ratings, test_ratings, item_count, None,
                                                           sample_count=batch_count, batch_size=batch_size):
                _batch_loss, _ = self.session.run([bprloss, train_op], feed_dict={u:batch[:,0], i:batch[:,1],
                                                                                  j:batch[:,2]})
                train_loss_vals.append(_batch_loss)
                duration = time.time() - epoch_start_time

                yield epoch, duration, np.mean(train_loss_vals)

    @classmethod
    def bpr_mf(cls, user_count, item_count, hidden_dim, lr=0.2, regulation_rate=0.0005, bias_reg=.01): # 类方法，不需要实例化这个类就可以用
        # model input
        u = tf.placeholder(tf.int32, [None])
        i = tf.placeholder(tf.int32, [None])
        j = tf.placeholder(tf.int32, [None])

        # model paramaters
        user_emb_w = tf.get_variable('user_emb_w', [user_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
        item_emb_w = tf.get_variable('item_emb_w', [item_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
        item_b = tf.get_variable('item_b', [item_count+1 ,1], initializer=tf.random_normal_initializer(0, 0.1))
        user_b = tf.get_variable('user_b', [user_count+1, 1], initializer=tf.random_normal_initializer(0, 0.1))

        #
        u_emb = tf.nn.embedding_lookup(user_emb_w, u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, i)
        j_emb = tf.nn.embedding_lookup(item_emb_w, j)
        i_b = tf.nn.embedding_lookup(item_b, i)
        j_b = tf.nn.embedding_lookup(item_b, j)

        # MF predict: u_i > u_j
        xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True)
        xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True)
        xuij = xui - xuj

        # AUC for one user
        mf_auc = tf.reduce_mean(tf.to_float(xuij > 0))
        tf.summary.scalar('user_auc', mf_auc)

        # 正则项
        l2_norm = tf.add_n([
            regulation_rate * tf.reduce_sum(tf.multiply(u_emb, u_emb)),
            regulation_rate * tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            regulation_rate * tf.reduce_sum(tf.multiply(j_emb, j_emb)),
            # reg for biases
            bias_reg * tf.reduce_sum(tf.multiply(i_b, i_b)),
            bias_reg / 10.0 * tf.reduce_sum(tf.multiply(j_b, j_b)),
        ])

        # 计算loss
        bprloss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))

        # 这里采用动态学习率
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, 400, 0.8, staircase=True)
        # .1 .... .001

        # 调用学习器，将loss和lr填充进去，构成train单元
        train_op = tf.train.AdamOptimizer().minimize(bprloss, global_step=global_step)
        return u, i, j, mf_auc, bprloss, train_op

    # 这里用auc来evaluate跑出来的结果
    def evaluate(self, eval_set, sample_size=200, cold_start=True):
        u, i, j, auc, loss, train_op = self.u, self.i, self.j, self.mf_auc, self.bprloss, self.train_op

        loss_vals = []
        auc_vals = []
        for uij in self.generate_user_eval_batch(self.corpus.user_items, eval_set, self.corpus.item_count,
                                                 self.corpus.item_dist, None, sample_size=sample_size, cold_start=cold_start):
            _loss, user_auc = self.session.run([loss, auc], feed_dict={u:uij[:,0], i:uij[:,1], j:uij[:,2]})
            loss_vals.append(_loss)
            auc_vals.append(user_auc)

        auc = np.mean(auc_vals)
        loss = np.mean(loss_vals)
        return auc, loss

    def generate_results(self):
        variable_names = [v.name for v in tf.trainable_variables()]
        values = self.session.run(variable_names)
        i =0
        for k, v in zip(variable_names, values):
            if(i>5):
                break
            print("variable: ",k)
            print("shape: ", v.shape)
            print(v)
            i+=1
        u1_dim = tf.expand_dims(values[0][0],0)
        u1_all = tf.matmul(u1_dim, values[1], transpose_b=True)
        result1 = self.session.run(u1_all)
        a = np.exp(10).astype('float64')
        a = tf.cast(tf.expand_dims([a],0), tf.float32)
        augment = tf.matmul(a, result1)
        result2 = self.session.run(augment)
        
        print(result2)
        print("以下是给id为'A3LKP6WPMP9UKX'用户的推荐：")
        p = np.squeeze(result2)
        #p[np.argsort((p)[:-5])]  = 0
        p1 = np.argsort((p))[-50:]
        for index in range(len(p1)):
            print(index, p1[index], p[p1[index]])
            
        return p1
    
        
        