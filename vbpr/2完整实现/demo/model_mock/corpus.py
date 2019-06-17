from collections import defaultdict
import numpy as np
import random
import csv

# 这个类是一个类似数据库一样的存在；建立数据结构，收集并组织其他类会用到的各种数据
# 1.stats: 统计user和item每个出现的次数
# 2.load_complex: 读取meta文件，并过滤掉一些评论数少于min的用户, 返回构建集合users, items, np.array(triples), brands, prices
# 3.load_simple: 不用读取meta文件，只用构建简单的user-item对，返回users, items, np.array(triples)
# 4.load_heuristics: 将image,meta的特征融合（调用了函数5，并设置self.image_features）
# 5.merge_image_features_and_meta：将meta中的price，brands和image_feature分别独热编码后，拼接起来，返回一个融合数组
# 6.load_image_features：将给定用户collection的image_features（raw文件）整合入字典形式
# 7.load_reviews：调用函数1和2，加入到self的参数中
# 8.load_images：调用函数6，加入到self参数中
# 9.load_data：调用函数7和函数8，加入到self参数中


class Corpus(object):
    #
    def __init__(self):
        super(Corpus, self).__init__()

    @staticmethod
    def stats(reviews): # 每个u和i出现的次数记录一下..
        user_dist = defaultdict(int)
        item_dist = defaultdict(int)

        # cores，user-item对和item-user对
        user_items = defaultdict(list)
        item_users = defaultdict(list)

        for review in reviews:
            u = review[0]
            i = review[1]
            user_dist[u]+=1
            item_dist[i]+=1
            user_items[u].append(i)
            item_users[i].append(u)

        return user_dist, item_dist, user_items, item_users

    @staticmethod
    def load_complex(path, user_min=5): # 加载product_description文件
        print("load_complex")
        # 从磁盘中读raw数据
        reviews = []
        with open(path, "r") as f:
            next(f)
            csvreader = csv.reader(f)
            for auid, asin, _, brand, price, _ in csvreader:
                reviews.append([auid, asin, brand, price])

        user_dist, item_dist, user_ratings, item_users = Corpus.stats(reviews)
        
        flag = 0
        # 基于user_dist过滤掉一些评论过少的用户
        reviews_reduced = []
        for auid, asin, brand, price in reviews:
            if user_dist[auid] >= user_min:
                if flag == 0:
                    print("first auid: "+auid)
                    flag = 1
                reviews_reduced.append([auid, asin, brand, price])

        users = {}
        items = {}
        brands = {}
        prices = {}
        user_count = 0
        item_count = 0
        triples = []

        for auid, asin, brand, price in reviews_reduced:
            if auid in users:
                u = users[auid]
            else:
                user_count += 1 #
                users[auid] = user_count
                u = user_count

            if asin in items:
                i = items[asin]
            else:
                item_count += 1
                items[asin] = item_count
                i = item_count

            brands[i] = brand
            if (price=="" or price=="\r\n" or price=='\n'):
                prices[i] = 0
            else:
                prices[i] = float(price.rstrip())

            triples.append([u, i])

        return users, items, np.array(triples), brands, prices

    @staticmethod
    def load_simple(path, user_min=5):
        print('load_simple')
        #load raw from disk
        reviews = []
        with open(path, 'r') as f:
            for line in f.readlines():
                auid, asin, _ = line.split(",",2)
                reviews.append([auid, asin])
        #stats
        user_dist, item_dist, user_ratings, item_users = Corpus.stats(reviews)

        #过滤掉那些少于min的用户
        reviews_reduced = []
        for auid, asin in reviews:
            if user_dist[auid] >= user_min:
                reviews_reduced.append([auid, asin])

        # 将其映射到sequential ids
        users = {}
        items = {}
        user_count = 0
        item_count = 0
        triples = []
        for auid, asin in reviews_reduced:
            if auid in users:
                u = users[auid]
            else:
                user_count += 1
                users[auid] = user_count
                u = user_count

            if asin in items:
                i = items[asin]
            else:
                item_count+=1
                items[asin] = item_count
                i = item_count
            triples.append([u,i])
        return users, items, np.array(triples)

    # 整合image features 和 meta features
    #
    def load_heuristics(self):
        image_features_plus = Corpus.merge_image_features_and_meta(self.brands, self.prices, self.image_features)
        # overwrite
        self.image_features = image_features_plus

    @staticmethod
    def merge_image_features_and_meta(brands, prices, image_features):
        # one-hot encode prices
        # 手动的价格独热编码
        # 注意这里的价格虽然是连续型特征，但是在这里需要按照离散值处理，因此用独热编码
        prices_features = {}
        prices_all = list(set(prices.values()))
        price_quant_level = 10 #正则化后散布在[0,10]区间
        price_max = float(max(prices.values()))
        for key, value in prices.items():
            prices_vec = np.zeros(price_quant_level + 1)
            idx = int(np.ceil(float(value) / (price_max / price_quant_level))) # ceil函数向上取整
            prices_vec[idx] = 1
            prices_features[key] = prices_vec

        #one-hot encode brands
        brands_features = {}
        brands_all = list(set(brands.values()))
        for key, value in brands.items():
            brands_vec = np.zeros(len(brands_all))
            brands_vec[brands_all.index(value)] = 1
            brands_features[key] = brands_vec

        a = prices_features
        b = brands_features

        # 将用户价格和brand的独热编码融合
        c = dict([(k, np.append(a[k], b[k])) for k in set(b) & set(a)])  # k算是索引

        # 将用户image特征数据和c向量融合
        f = image_features
        image_features_p = dict([(k, np.append(c[k], f[k])) for k in set(c) & set(f)])

        return image_features_p

    NORM_FACTOR = 58.388599

    @staticmethod
    def load_image_features(path, items):
        count = 0
        count_item = 0
        flag = 0
        image_features = {}
        f = open(path, 'rb') #加b是因为这是一个csv/txt文件

        # j = 0
        # for key in items.keys():
        #     print("item_key: ",key)
        #     j += 1
        #     if (j ==10):
        #         break

        while(flag<2):
            asin = f.read(10)
            if asin=="": break
            features_bytes = f.read(16384)  # 4*4096=16KB,
            asin = str(asin).replace("b",'').split("'")[1]
            if asin in items.keys():
                features = np.fromstring(features_bytes, dtype=np.float32)/Corpus.NORM_FACTOR
                iid = items[asin]
                image_features[iid] = features
                count_item += 1
            #print("asin:", asin)
            if count%10000000 == 0:
                print("count_read, ", count, "\t count_item, ",count_item)
                flag += 1
            count+=1

        return image_features



    def load_reviews(self, path, user_min):
        print("loading dataset from: ", path)
        uesrs, items, reviews_all, brands, prices = Corpus.load_complex(path, user_min=user_min)

        print("generate stats...")
        user_dist, item_dist, train_ratings, item_users = Corpus.stats(reviews_all)

        user_count = len(train_ratings)
        item_count = len(item_users)
        reviews_count = len(reviews_all)
        print("user_count:",user_count," item_count:", item_count, " reviews_count", reviews_count)

        self.users = item_users
        self.items = items
        self.reviews = reviews_all
        self.brands = brands
        self.prices = prices

        self.user_dist = user_dist
        self.item_dist = item_dist
        self.user_items = train_ratings
        self.item_users = item_users

        self.user_count = len(self.user_items)
        self.item_count = len(self.item_users)

    def load_images(self, path, items):
        print("laoding images features from ...",path)
        self.image_features = Corpus.load_image_features(path, items)

        print("提取iamge feature count:",len(self.image_features))

    def load_data(self, reviews_path, images_path, user_min, item_min):
        #加载数据
        self.load_reviews(reviews_path, user_min)
        if images_path:
            self.load_images(images_path, self.items)

if __name__ == "__main__":
    import os
    data_dir = os.path.join("data", "amzn")
    simple_path = os.path.join(data_dir, '1reviews_Women_ALL_scraped.csv')

    user, items, reviews, brands, prices = Corpus.load_complex(simple_path)
    image_features = Corpus.load_image_features("data/amzn/1image_features_Women.b", items)
    image_features_plus = Corpus.merge_image_features_and_meta(brands, prices, prices, image_features)








