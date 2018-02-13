# -*- coding: utf-8 -*-
import numpy as np
import pickle
save_path = '/home/silence/proj/'


if __name__ == '__main__':
    # 加载文件名列表
    data_names_file = open(save_path+'data-imagenames.pkl', 'rb')
    data_names = pickle.load(data_names_file)
    ref_names_file = open(save_path+'ref-imagenames.pkl', 'rb')
    ref_names = pickle.load(ref_names_file)
    num_of_ref = len(ref_names)
    # 加载特征文件
    data_style_features_file = open(save_path+'data-style-features.pkl', 'rb')
    data_style_features = pickle.load(data_style_features_file)
    data_luminance = data_style_features['luminance_features']
    data_mu = data_style_features['color_mu']
    data_cov = data_style_features['color_cov']
    ref_style_features_file = open(save_path+'ref-style-features.pkl', 'rb')
    ref_style_features = pickle.load(ref_style_features_file)
    ref_luminance = ref_style_features['luminance_features']
    ref_mu = ref_style_features['color_mu']
    ref_cov = ref_style_features['color_cov']
    # 加载聚类文件
    cluster_index_file = open(save_path+'index.pkl', 'rb')
    cluster_index = pickle.load(cluster_index_file)
    num_of_clusters = len(cluster_index)
    score = np.zeros((num_of_clusters,num_of_ref))
