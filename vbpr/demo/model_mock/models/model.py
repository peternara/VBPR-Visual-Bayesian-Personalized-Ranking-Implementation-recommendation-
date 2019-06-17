import random
import numpy
import tensorflow as tf
from datetime import datetime

class Model(object):
    # 1. 初始化session,
    # 网络搭建周边的准备工作做好，比如：保存中间数据，构建数据集
    # 关联： corpus

    def __init__(self, corpus, session):
        self.corpus = corpus
        self.session = session

        self.merged = tf.summary.merge_all() # 将全部的summary合并起来，并保存，以便在之后tensorboard显示
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.train_writer = tf.summary.FileWriter('logs/train/run-%s'%now, self.session.graph)  #创建一个train可视化输出文件
        self.test_writer = tf.summary.FileWriter('logs/test/run-%s'%now, self.session.graph)  #创建一个test数据可视化输出文件
        self.saver = tf.train.Saver()  #创建一个模型参数保存的对象
        #ckpt = tf.train.get_checkpoint_state('logs/')
        self.session.run(tf.global_variables_initializer()) # 初始化所有参数
        #if ckpt != '':
            #self.restore()
        self.val_ratings, self.test_ratings = self.generate_val_and_test()

    def generate_val_and_test(self):
        # 对于每个用户，随机选择一个rating放入test集
        user_test = dict()
        user_val = dict()
        for u,i_list in self.corpus.user_items.items():
            samples = random.sample(i_list, 2)
            user_test[u] = samples[0]
            user_val[u] = samples[1]
        return user_val, user_test

    # 将这个模型保存到指定路径
    def save(self):
        self.saver.save(self.session, 'logs/')

    def restore(self):
        self.saver.restore(self.session, "logs/")
        print("reloaded the parameters...")

    def train(self):
        raise Exception("Not implemented yet!")

    def evaluate(self):
        raise Exception("Not implemented yet")

    def export(filename):
        raise Exception("Not implemented yet")

    def generate_results(self):
        raise Exception("Not implemented yet")

    def generate_user_eval_batch(self,  user_items, test_ratings, item_count, item_dist, image_features,
                                 sample_size=3000, neg_sample_size=1000, cold_start=False):
        # 留一验证法
        for u in random.sample(test_ratings.keys(), sample_size):  # 正态分布取样
            t = []
            ilist = []
            jlist = []

            i = test_ratings[u]
            # 检查是否有i对应的图片，有时候没有
            if image_features and i not in image_features:
                continue

            # 为冷启动筛选
            if cold_start and item_dist[i] > 5:
                continue

            for _ in range(neg_sample_size):
                # range和xrange的区别：用法与range完全相同，所不同的是生成的不是一个数组，而是一个生成器。
                # 要生成很大的数字序列的时候，用xrange会比range性能优很多，因为不需要一上来就开辟一块很大的内存空间
                j = random.randint(1, item_count)
                if j != test_ratings[u] and not (j in user_items[u]):
                    # 找到那些不在train或test集中的negative项
                    # 有时候一个商品并没有对应的图片
                    if image_features:
                        try:
                            image_features[i]
                            image_features[j]
                        except KeyError:
                            continue

                    t.append([u, i, j])

                    if image_features:
                        ilist.append(image_features[i])
                        jlist.append(image_features[j])

            if image_features:
                yield numpy.asarray(t), numpy.vstack(tuple(ilist)), numpy.vstack(tuple(jlist))
            else:
                yield numpy.asarray(t)

if __name__ == '__main__':
    import os
    import corpus
    print("Loading dataset...")
    data_dir = os.path.join("data", "amzn")
    simple_path = os.path.join(data_dir, "review_Women_5.txt")







