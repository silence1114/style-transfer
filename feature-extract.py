# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from matplotlib import pyplot
# 设置当前的工作环境在caffe下
caffe_root = '/home/silence/caffe/caffe/'
# 把caffe/python也添加到当前环境
sys.path.insert(0, caffe_root + 'python')
import caffe
# 更换工作目录
os.chdir(caffe_root)
caffe.set_mode_cpu()
#caffe.set_mode_gpu()
# 设置网络结构
net_file = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
# 添加训练之后的参数
caffe_model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# 将上面两个变量作为参数构造一个net
net = caffe.Net(net_file, #定义模型结构
                caffe_model, #包含模型训练权值
                caffe.TEST) #使用测试模式(不执行dropout)
# 加载Imagenet的图像均值文件
mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
''' 
matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB  
caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转换
'''
# channel放到前面
transformer.set_transpose('data',(2,0,1))
# 对于每个通道，都减去BGR的均值像素值
transformer.set_mean('data',np.load(mean_file).mean(1).mean(1))
# 图片像素放大到[0-255]
transformer.set_raw_scale('data',255)
# RGB -> BGR 转换
transformer.set_channel_swap('data',(2,1,0))

# 加载图片
im = caffe.io.load_image('/home/silence/proj/photos/23497071398.jpg')
# 用之前的设定处理图片 
transformed_image = transformer.preprocess('data',im)
# 将图像数据拷贝到为net分配的内存中
net.blobs['data'].data[...] = transformed_image
# 网络向前传播
output = net.forward()
# 结果（属于某个类别的概率值）
# output_pro的shape中有对于1000个object相似的概率 
output_prob = output['prob'][0]  #batch中第一张图像的概率值   
# 加载imagenet标签
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
labels = np.loadtxt(labels_file,str,delimiter='\t')
print('output label:', labels[output_prob.argmax()])
# 前5名的概率和类别
top_inds = output_prob.argsort()[::-1][:5]    
print ('probabilities and labels:')
for top_i in top_inds:
    print(output_prob[top_i],labels[top_i])  
'''
# 前5名的类别
top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
for i in np.arange(top_k.size):
    print(top_k[i],labels[top_k[i]])
'''



