# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
# 加载特征数据
save_path = '/home/silence/proj/'
file_feature = open(save_path+'features.pkl', 'rb')
features = pickle.load(file_feature)
# 聚类
num_clusters = 3
km_cluster = KMeans(n_clusters=num_clusters,n_jobs=-1)
km_cluster.fit(features)
file_cluster = open(save_path+'clusters.pkl', 'wb')
pickle.dump(km_cluster.labels_,file_cluster)
# 存储结果
labels = km_cluster.labels_.copy()
file_names = open(save_path+'photonames.pkl','rb')
photonames = np.array(pickle.load(file_names))
index_list = []
names_list = []
for label in range(num_clusters):
    index = np.where(labels == label)[0]
    index.sort()
    index_list.append(index.copy())
    names_list.append(photonames[index].copy())
file_index = open(save_path+'index.pkl','wb')
pickle.dump(index_list,file_index)
clustered_filenames = open(save_path+'clusteredNames.pkl','wb')
pickle.dump(names_list,clustered_filenames)



