# -*- coding: utf-8 -*-
import os
import numpy as np
from skimage import io, color
save_path = '/home/silence/proj/style_ref_demo/'
def extract_style_feature(filename):
    # 加载rgb图像
    rgb = io.imread(save_path+filename)
    # 转换到CIELab色域
    lab = color.rgb2lab(rgb)
    # 归一化，L,a,b各分量原本范围:L[0,100], a,b[-128,127]
    lab[:,:,0] = lab[:,:,0]/100
    lab[:,:,1:3] = (lab[:,:,1:3]+128)/256
    #print(np.array(lab[:,:,0]).flatten().tolist())
    
    

if __name__ == '__main__':
    extract_style_feature('23474274708.jpg')


