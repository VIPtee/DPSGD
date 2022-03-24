# -*- coding: utf-8 -*-
import numpy as np;
import tensorflow as tf;
# import tensorflow._api.v2.compat.v1 as tf
# tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
import xlsxwriter
from tensorflow.python.framework import ops;
from tensorflow.examples.tutorials.mnist import input_data;
#import tensorflow.keras as keras
import argparse;
import numpy as np
import math
import random
import scipy.integrate as integrate
import scipy.stats
import mpmath as mp
from gaussian_moments import *;
from tensorflow.python.platform import flags
from datetime import datetime
import time
from tensorflow.python.training import optimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
import accountant, utils
from dpSGD.dpSGD_MNIST.gaussian_moments import compute_log_moment, get_privacy_spent
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #使用第1块GPU

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True);
# mnist = keras.datasets.mnist.load_data()

def idle():
    return


# compute sigma using strong composition theory given epsilon
# 在给定epsilon的情况下使用强成分理论计算sigma
def compute_sigma(epsilon, delta):
    return 1 / epsilon * np.sqrt(np.log(2 / math.pi / np.square(delta)) + 2 * epsilon)


# compute sigma using moment accountant given epsilon
# 使用给定epsilon的矩会计来计算sigma
def comp_sigma(q, T, delta, epsilon):
    c_2 = 4 * 1.26 / (0.01 * np.sqrt(10000 * np.log(100000)))  # c_2 = 1.485
    return c_2 * q * np.sqrt(T * np.log(1 / delta)) / epsilon


# compute epsilon using abadi's code given sigma
# 使用给定sigma的abadi代码计算epsilon
def comp_eps(lmbda, q, sigma, T, delta):
    lmbds = range(1, lmbda + 1)
    log_moments = []
    for lmbd in lmbds:  # eg. range(1,5)   [1, 2, 3, 4]
        log_moment = compute_log_moment(q, sigma, T, lmbd)
        log_moments.append((lmbd, log_moment))

    eps, delta = get_privacy_spent(log_moments, target_delta=delta)
    return eps


# 有许多权重偏置要创建，先定义俩留着用
# 截尾正态分布，保留[mean-2*stddev, mean+2*stddev]范围内的随机数。用于初始化所有的权值，用做卷积核。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1);  # 标准差0.1
    return tf.Variable(initial);


# 创建常量0.1；用于初始化所有的偏置项，即b，用作偏置。
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape);  # 偏置增加小正值防止死亡节点
    return tf.Variable(initial);


# 定义一个函数，用于构建卷积层；
# x为input；w为卷积核；strides是卷积时图像每一维的步长；padding为不同的卷积方式；
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME');  # 卷积的输出输入保持同样的尺寸  # strides[图片，长，宽，channel]


# 定义一个函数，用于构建池化层，池化层是为了获取特征比较明显的值，一般会取最大值max，有时也会取平均值mean。
# ksize=[1,2,2,1]：shape为[batch，height，width， channels]设为1个池化，池化矩阵的大小为2*2,有1个通道。
# strides是表示步长[1,2,2,1]:水平步长为2，垂直步长为2，strides[0]与strides[3]皆为1。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME');


FLAGS = None;

# target_eps = [0.125,0.25,0.5,1,2,4,8]
target_eps = [8];

# update_flag=0  #0

lot_size=600

global_grad=0

# now_gw_W1 = None
# now_gb1 = None
# now_gw_W2 = None
# now_gb2 = None
# now_gw_W3 = None
# now_gb3 = None
now_gw_Wf1 = None
now_gbf1 = None
# now_gw_Wf1_2 = None
# now_gbf1_2 = None
now_gw_Wf2 = None
now_gbf2 = None

constant_value_1 = tf.constant(1.0)

def compute_new_gradients(y_, y_conv, batch_size, W_fc1, b_fc1, W_fc2, b_fc2,  clip_bound, sigma, sensitivity):
  sum_g_W_fc1 = None
  sum_g_b_fc1 = None
  sum_g_W_fc2 = None
  sum_g_b_fc2 = None

  for batch_i in range(batch_size):
      # idx = tf.constant([batch_i])
      # y_i = tf.gather(y_, idx)
      # y_conv_i = tf.gather(y_conv, idx)
      # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_i, logits=y_conv_i)
      # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_[batch_i], logits=y_conv[batch_i]))
      g_W_fc1 = tf.gradients(cross_entropy, W_fc1)[0]  # gradient of wb
      g_b_fc1 = tf.gradients(cross_entropy, b_fc1)[0]  # gradient of wb
      g_W_fc2 = tf.gradients(cross_entropy, W_fc2)[0]  # gradient of wb
      g_b_fc2 = tf.gradients(cross_entropy, b_fc2)[0]  # gradient of wb

      now_gl2 = tf.sqrt(2.0 * (tf.nn.l2_loss(g_W_fc1)+tf.nn.l2_loss(g_b_fc1)+tf.nn.l2_loss(g_W_fc2)+tf.nn.l2_loss(g_b_fc2)))

      # now_gl2_w1 = tf.sqrt(2.0 * (tf.nn.l2_loss(g_W_fc1)))
      # now_gl2_b1 = tf.sqrt(2.0 * (tf.nn.l2_loss(g_b_fc1)))
      # now_gl2_w2 = tf.sqrt(2.0 * (tf.nn.l2_loss(g_W_fc2)))
      # now_gl2_b2 = tf.sqrt(2.0 * (tf.nn.l2_loss(g_b_fc2)))

      # if(now_gl2>clip_bound):
      g_W_fc1 = g_W_fc1 / tf.maximum(constant_value_1, now_gl2 / clip_bound)   #1 91.26% 1 91.33% 4 88.5% 4 89.39% 0.5  89.91%  1 91.35%   1   91.61%  0.1  81.4%  1   91.61%
      g_b_fc1 = g_b_fc1 / tf.maximum(constant_value_1, now_gl2 / clip_bound) #4        1        4        1       0.5          0.5       0.1            0.1      0.05
      g_W_fc2 = g_W_fc2 / tf.maximum(constant_value_1, now_gl2 / clip_bound)   #1        1        4        4       0.5          1         1            0.1         1
      g_b_fc2 = g_b_fc2 / tf.maximum(constant_value_1, now_gl2 / clip_bound) #4        1        4        1       0.5          0.5       0.1            0.1      0.05
      # g_W_fc1 = g_W_fc1 / tf.maximum(constant_value_1, now_gl2_w1 / 1)   #1    1  89.84%    4            4      0.5    1    1     0.1    1
      # g_b_fc1 = g_b_fc1 / tf.maximum(constant_value_1, now_gl2_b1 / 1) #4        1        4        1       0.5          0.5       0.1            0.1      0.05
      # g_W_fc2 = g_W_fc2 / tf.maximum(constant_value_1, now_gl2_w2 / 1)   #1        1        4        4       0.5          1         1            0.1         1
      # g_b_fc2 = g_b_fc2 / tf.maximum(constant_value_1, now_gl2_b2 / 1) #4        1        4        1       0.5          0.5       0.1            0.1      0.05

      # now_gl2 = tf.sqrt(2.0 * (tf.nn.l2_loss(g_W_fc1) + tf.nn.l2_loss(g_b_fc1)+tf.nn.l2_loss(g_W_fc2) + tf.nn.l2_loss(g_b_fc2)))
      #
      # # if(now_gl2>clip_bound):
      # g_W_fc1 = g_W_fc1 / tf.maximum(constant_value_1, now_gl2 / clip_bound)
      # g_b_fc1 = g_b_fc1 / tf.maximum(constant_value_1, now_gl2 / clip_bound)
      # g_W_fc2 = g_W_fc2 / tf.maximum(constant_value_1, now_gl2 / clip_bound)
      # g_b_fc2 = g_b_fc2 / tf.maximum(constant_value_1, now_gl2 / clip_bound)
      # else:



      if(batch_i==batch_size-1):
        sum_g_W_fc1 += g_W_fc1# + tf.random_normal(shape=tf.shape(g_W_fc1), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size
        sum_g_b_fc1 += g_b_fc1# + tf.random_normal(shape=tf.shape(g_b_fc1), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size
        sum_g_W_fc2 += g_W_fc2# + tf.random_normal(shape=tf.shape(g_W_fc2), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size
        sum_g_b_fc2 += g_b_fc2# + tf.random_normal(shape=tf.shape(g_b_fc2), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size
        # sum_g_wb = sum_g_wb
        return sum_g_W_fc1,sum_g_b_fc1,sum_g_W_fc2,sum_g_b_fc2
      elif(batch_i==0):
        sum_g_W_fc1= g_W_fc1# + tf.random_normal(shape=tf.shape(g_W_fc1), mean=0.0, stddev=sigma * sensitivity, dtype=tf.float32)/batch_size
        sum_g_b_fc1= g_b_fc1# + tf.random_normal(shape=tf.shape(g_b_fc1), mean=0.0, stddev=sigma * sensitivity, dtype=tf.float32)/batch_size
        sum_g_W_fc2= g_W_fc2# + tf.random_normal(shape=tf.shape(g_W_fc2), mean=0.0, stddev=sigma * sensitivity, dtype=tf.float32)/batch_size
        sum_g_b_fc2= g_b_fc2# + tf.random_normal(shape=tf.shape(g_b_fc2), mean=0.0, stddev=sigma * sensitivity, dtype=tf.float32)/batch_size
      else:
        sum_g_W_fc1 += g_W_fc1# + tf.random_normal(shape=tf.shape(g_W_fc1), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size
        sum_g_b_fc1 += g_b_fc1# + tf.random_normal(shape=tf.shape(g_b_fc1), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size
        sum_g_W_fc2 += g_W_fc2# + tf.random_normal(shape=tf.shape(g_W_fc2), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size
        sum_g_b_fc2 += g_b_fc2# + tf.random_normal(shape=tf.shape(g_b_fc2), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/batch_size

# def compute_new_gradients(y_, y_conv, batch_size, wb, clip_bound, sigma, sensitivity):
#   sum_g_wb = None
#   for batch_i in range(batch_size):
#       # idx = tf.constant([batch_i])
#       # y_i = tf.gather(y_, idx)
#       # y_conv_i = tf.gather(y_conv, idx)
#       # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_i, logits=y_conv_i)
#       # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#       cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_[batch_i], logits=y_conv[batch_i])
#       g_wb = tf.gradients(cross_entropy, wb)[0]  # gradient of wb
#       g_wb = tf.clip_by_norm(g_wb, clip_bound)
#
#       if(batch_i==batch_size-1):
#         sum_g_wb += g_wb + tf.random_normal(shape=tf.shape(g_wb), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)
#         sum_g_wb = sum_g_wb
#         return sum_g_wb
#       elif(batch_i==0):
#         sum_g_wb= g_wb + tf.random_normal(shape=tf.shape(g_wb), mean=0.0, stddev=sigma * sensitivity, dtype=tf.float32)
#       else:
#         sum_g_wb += g_wb + tf.random_normal(shape=tf.shape(g_wb), mean=0.0, stddev=sigma * sensitivity, dtype=tf.float32)



def main(_):
    # global update_flag
    clip_bound = 4  #the clip bound of the gradients'
    clip_bound_2 = 1 / 1.5  # 'the clip bound for r_kM'

    small_num = 1e-5  # 'a small number'
    large_num = 1e5  # a large number'
    num_images = 60000  # 'number of images N'
    num_LayerNodes = 1000
    batch_size = 10  # 'batch_size L'
    sample_rate = 600 / 60000  # 'sample rate q = L / N'
    # num_steps = 160000  # 'number of steps T = E * N / L = E / q'
    num_steps = 436*6000
    num_epoch = 24  # 'number of epoches E'

    sigma = 2.0  # 'sigma'
    delta = 1e-5  # 'delta'

    lambd = 1e3  # 'exponential distribution parameter'

    iterative_clip_step = 2  # 'iterative_clip_step'

    clip = 1  # 'whether to clip the gradient'
    noise = 0  # 'whether to add noise'
    redistribute = 0  # 'whether to redistribute the noise'

    D = 60000;

    # global now_gw_W1
    # global now_gb1
    # global now_gw_W2
    # global now_gb2
    # global now_gw_W3
    # global now_gb3
    global now_gw_Wf1
    global now_gbf1
    # global now_gw_Wf1_2
    # global now_gbf1_2
    global now_gw_Wf2
    global now_gbf2

    '''from tensorflow.examples.tutorials.mnist import input_data;
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True);'''

    # tf.InteractiveSession():它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)
    # 构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。
    sess = tf.InteractiveSession();  #

    # Create the model
    # 第一个卷积层+池化
    x = tf.placeholder(tf.float32, [None,784]);  # 定义实际x与y的值。   # placeholder中shape是参数的形状，默认为none，即一维数据，[2,3]表示为两行三列；[none，3]表示3列，行不定。
    # x_image = tf.reshape(x, [-1, 28, 28, 1]);  # 在reshape方法中-1维度表示为自动计算此维度，将x按照28*28进行图片转换，转换成一个大包下一个小包中28行28列的四维数组；
    # # W_conv1 = weight_variable([5, 5, 1, 32]);# 构建一定形状的截尾正态分布，用做第一个卷积核； # 卷积核大小：5x5  通道数：1  卷积核数目：32个
    # W_conv1 = weight_variable([5, 5, 1, 32]);  # 64改成32了
    # # b_conv1 = bias_variable([32]);# 构建一维的偏置量。
    # b_conv1 = bias_variable([32]);
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1);  # 将卷积后的结果进行relu函数运算，通过激活函数进行激活。
    # h_pool1 = max_pool_2x2(h_conv1);  # 将激活函数之后的结果进行池化，降低矩阵的维度。
    #
    # # 第二个卷积层+池化
    # #W_conv2 = weight_variable([5, 5, 32, 64]);# 构建第二个卷积核；
    # W_conv2 = weight_variable([5, 5, 32, 64]);
    # b_conv2 = bias_variable([64]);# 第二个卷积核的偏置；
    # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2);# 第二次进行激活函数运算；
    # h_pool2 = max_pool_2x2(h_conv2);# 第二次进行池化运算，输出一个2*2的矩阵，步长是2*2；

    # # 第三个卷积层+池化
    # W_conv3 = weight_variable([5, 5, 64, 64]);
    # b_conv3 = bias_variable([64]);# 第二个卷积核的偏置；
    # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3);# 第二次进行激活函数运算；
    # h_pool3 = max_pool_2x2(h_conv3);# 第二次进行池化运算，输出一个2*2的矩阵，步长是2*2；

    # 连一个全连接层
    W_fc1 = weight_variable([784, num_LayerNodes]);#2层 构建新的卷积核，用来进行全连接层运算，通过这个卷积核，将最后一个池化层的输出数据转化为一维的向量1*1024。
    # W_fc1 = weight_variable([14 * 14 * 32, 1024]);  # 构建新的卷积核，用来进行全连接层运算，通过这个卷积核，将最后一个池化层的输出数据转化为一维的向量1*1024。
    # W_fc1 = weight_variable([14 * 14 * 64, 1024]);  # 1层
    # W_fc1 = weight_variable([4 * 4 * 64, 1024]); #3层

    b_fc1 = bias_variable([num_LayerNodes]);  # 构建1*1024的偏置；

    h_pool2_flat = tf.reshape(x, [-1, 784]);#2层 对 h_pool2第二个池化层结果进行变形。
    # h_pool1_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 32]);  # 对 h_pool2第二个池化层结果进行变形。 老1
    # h_pool1_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 64]);  # 新1
    # h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64]);#3层

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1);# 2层 将矩阵相乘，并进行relu函数的激活。
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1);  # 1层将矩阵相乘，并进行relu函数的激活。
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1);#3层


    # 第二个全连接层
    # W_fc1_2 = weight_variable([4096, 4096]);#2层 构建新的卷积核，用来进行全连接层运算，通过这个卷积核，将最后一个池化层的输出数据转化为一维的向量1*1024。
    # b_fc1_2 = bias_variable([4096]);  # 构建1*1024的偏置；
    # h_fc1_2 = tf.nn.relu(tf.matmul(h_fc1, W_fc1_2) + b_fc1_2);# 2层 将矩阵相乘，并进行relu函数的激活。



    # Dropout层
    keep_prob = tf.placeholder(tf.float32);  # 定义一个占位符。
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);  # 是防止过拟合的，使输入tensor中某些元素变为0，其他没变为零的元素变为原来的1/keep_prob大小，

    # Softmax层

    # 形成防止过拟合之后的矩阵。
    W_fc2 = weight_variable([num_LayerNodes,10]);
    b_fc2 = bias_variable([10]);
    # y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2;
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2;

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10]);

    # d = 25*10 + 25*7*7*64 + 5*5*32*64 + 5*5*32; # number of parameters 参数数量
    # M = d

    priv_accountant = accountant.GaussianMomentsAccountant(D)
    privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], sigma, batch_size)  # 隐私累计消耗

    # sess.run(tf.initialize_all_variables())

    now_learning_rate = tf.placeholder(tf.float32, None)
    opt = GradientDescentOptimizer(learning_rate=now_learning_rate)
    # opt = GradientDescentOptimizer(learning_rate=0.05)

    # tf.nn.softmax_cross_entropy_with_logits这个函数内部自动计算softmax，然后再计算交叉熵代价函数，也就是说logits必须是没有经过tf.nn.softmax函数处理的数据，否则导致训练结果有问题。建议编程序时使用这个函数，而不必自己编写交叉熵代价函数。

    # batch_i = 0
    #
    # for batch_i in range(5):
    #     idx = tf.constant([batch_i])
    #     a = tf.gather(y_, idx)
    #     b = tf.gather(y_conv, idx)

    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv));
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    # train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy);
    # train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)

      # compute gradient 计算梯度
      # conv卷积层


    # gw_W1 = tf.gradients(cross_entropy, W_conv1)[0]  # gradient of W1
    # gb1 = tf.gradients(cross_entropy, b_conv1)[0]  # gradient of b1
    #
    # gw_W2 = tf.gradients(cross_entropy,W_conv2)[0] # gradient of W2
    # gb2 = tf.gradients(cross_entropy,b_conv2)[0] # gradient of b2
    #
    # # gw_W3 = tf.gradients(cross_entropy,W_conv3)[0] # gradient of W3
    # # gb3 = tf.gradients(cross_entropy,b_conv3)[0] # gradient of b3
    #
    # gw_Wf1 = tf.gradients(cross_entropy, W_fc1)[0]  # gradient of W_fc1
    # gbf1 = tf.gradients(cross_entropy, b_fc1)[0]  # gradient of b_fc1
    #
    # gw_Wf1_2 = tf.gradients(cross_entropy, W_fc1_2)[0]  # gradient of W_fc1
    # gbf1_2 = tf.gradients(cross_entropy, b_fc1_2)[0]  # gradient of b_fc1
    #
    # gw_Wf2 = tf.gradients(cross_entropy, W_fc2)[0]  # gradient of W_fc2
    # gbf2 = tf.gradients(cross_entropy, b_fc2)[0]  # gradient of b_fc2

    # clip gradient
    # gw_W1 = tf.clip_by_norm(gw_W1, clip_bound)
    # gb1 = tf.clip_by_norm(gb1, clip_bound)
    #
    # gw_W2 = tf.clip_by_norm(gw_W2,clip_bound)
    # gb2 = tf.clip_by_norm(gb2, clip_bound)
    #
    # # gw_W3 = tf.clip_by_norm(gw_W3, clip_bound)
    # # gb3 = tf.clip_by_norm(gb3, clip_bound)
    #
    # gw_Wf1 = tf.clip_by_norm(gw_Wf1, clip_bound)
    # gbf1 = tf.clip_by_norm(gbf1, clip_bound)
    #
    # gw_Wf1_2 = tf.clip_by_norm(gw_Wf1_2, clip_bound)
    # gbf1_2 = tf.clip_by_norm(gbf1_2, clip_bound)
    #
    # gw_Wf2 = tf.clip_by_norm(gw_Wf2, clip_bound)
    # gbf2 = tf.clip_by_norm(gbf2, clip_bound)
    # # sigma = FLAGS.sigma # when comp_eps(lmbda,q,sigma,T,delta)==epsilon

    # sensitivity = 2 * FLAGS.clip_bound #adjacency matrix with one tuple different
    sensitivity = clip_bound  # adjacency matrix with one more tuple 具有多个元组的邻接矩阵

    # shape: 输出张量的形状，必选
    # mean: 正态分布的均值，默认为0
    # stddev: 正态分布的标准差，默认为1.0
    # dtype: 输出的类型，默认为tf.float32
    # gw表示梯度
    #
    # flag=0

    # gw_W1 = compute_new_gradients(y_, y_conv, batch_size, W_conv1, clip_bound, sigma, sensitivity)
    # gb1 = compute_new_gradients(y_, y_conv, batch_size, b_conv1, clip_bound, sigma, sensitivity)
    # gw_W2 = compute_new_gradients(y_, y_conv, batch_size, W_conv2, clip_bound, sigma, sensitivity)
    # gb2 = compute_new_gradients(y_, y_conv, batch_size, b_conv2, clip_bound, sigma, sensitivity)
    # # gw_W3 = compute_new_gradients(y_, y_conv, batch_size, W_conv3, clip_bound, sigma, sensitivity)
    # # gb3 = compute_new_gradients(y_, y_conv, batch_size, b_conv3, clip_bound, sigma, sensitivity)
    gw_Wf1,gbf1,gw_Wf2,gbf2 = compute_new_gradients(y_, y_conv, batch_size, W_fc1, b_fc1, W_fc2, b_fc2, clip_bound, sigma, sensitivity)
    # gw_Wf1= compute_new_gradients(y_, y_conv, batch_size, W_fc1, clip_bound, sigma, sensitivity)
    # gbf1 = compute_new_gradients(y_, y_conv, batch_size, b_fc1, clip_bound, sigma, sensitivity)
    # # gw_Wf1_2 = compute_new_gradients(y_, y_conv, batch_size, W_fc1_2, clip_bound, sigma, sensitivity)
    # # gbf1_2 = compute_new_gradients(y_, y_conv, batch_size, b_fc1_2, clip_bound, sigma, sensitivity)
    # gw_Wf2 = compute_new_gradients(y_, y_conv, batch_size, W_fc2, clip_bound, sigma, sensitivity)
    # gbf2 = compute_new_gradients(y_, y_conv, batch_size, b_fc2, clip_bound, sigma, sensitivity)

    # gw_W1_grad_0 = tf.constant(0, shape=tf.shape(gw_W1))
    # gb1_grad_0 = tf.constant(0, shape=tf.shape(gb1))
    # gw_W2_grad_0 = tf.constant(0, shape=tf.shape(gw_W2))
    # gb2_grad_0 = tf.constant(0, shape=tf.shape(gb2))
    # gw_Wf1_grad_0 = tf.constant(0, shape=tf.shape(gw_Wf1))
    # gbf1_grad_0 = tf.constant(0, shape=tf.shape(gbf1))
    # gw_Wf1_2_grad_0 = tf.constant(0, shape=tf.shape(gw_Wf1_2))
    # gbf1_2_grad_0 = tf.constant(0, shape=tf.shape(gbf1_2))
    # gw_Wf2_grad_0 = tf.constant(0, shape=tf.shape(gw_Wf2))
    # gbf2_grad_0 = tf.constant(0, shape=tf.shape(gbf2))

        # gw_W1 += tf.random_normal(shape=tf.shape(gw_W1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gb1 += tf.random_normal(shape=tf.shape(gb1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gw_W2 += tf.random_normal(shape=tf.shape(gw_W2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gb2 += tf.random_normal(shape=tf.shape(gb2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gw_W3 += tf.random_normal(shape=tf.shape(gw_W3), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gb3 += tf.random_normal(shape=tf.shape(gb3), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gw_Wf1 += tf.random_normal(shape=tf.shape(gw_Wf1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gbf1 += tf.random_normal(shape=tf.shape(gbf1), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gw_Wf2 += tf.random_normal(shape=tf.shape(gw_Wf2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)
        # gbf2 += tf.random_normal(shape=tf.shape(gbf2), mean=0.0, stddev = (sigma * sensitivity)**2, dtype=tf.float32)

    # 2层
    # train_step = opt.apply_gradients([(gw_W1,W_conv1),(gb1,b_conv1),(gw_W2,W_conv2),(gb2,b_conv2),(gw_Wf1,W_fc1),(gbf1,b_fc1),(gw_Wf2,W_fc2),(gbf2,b_fc2)]);
    # 1层
    # if(update_flag == 0):
    #     # train_step = opt.apply_gradients(
    #     #     [(gw_W1_grad_0, W_conv1), (gb1_grad_0, b_conv1), (gw_W2_grad_0, W_conv2), (gb2_grad_0, b_conv2), (gw_Wf1_grad_0, W_fc1), (gbf1_grad_0, b_fc1),
    #     #      (gw_Wf1_2_grad_0, W_fc1_2), (gbf1_2_grad_0, b_fc1_2), (gw_Wf2_grad_0, W_fc2), (gbf2_grad_0, b_fc2)]);
    #
    # elif(update_flag == 1):
    # update_gw_W1 = tf.placeholder(tf.float32, None)
    # update_gb1 = tf.placeholder(tf.float32, None)
    # update_gw_W2 = tf.placeholder(tf.float32, None)
    # update_gb2 = tf.placeholder(tf.float32, None)
    update_gw_Wf1 = tf.placeholder(tf.float32, None)
    update_gbf1 = tf.placeholder(tf.float32, None)
    # update_gw_Wf1_2 = tf.placeholder(tf.float32, None)
    # update_gbf1_2 = tf.placeholder(tf.float32, None)
    update_gw_Wf2 = tf.placeholder(tf.float32, None)
    update_gbf2 = tf.placeholder(tf.float32, None)

    # train_step = opt.apply_gradients([(update_gw_W1, W_conv1), (update_gb1, b_conv1),(update_gw_W2,W_conv2),(update_gb2,b_conv2), (update_gw_Wf1, W_fc1),
    #                                   (update_gbf1, b_fc1),(update_gw_Wf1_2, W_fc1_2), (update_gbf1_2, b_fc1_2), (update_gw_Wf2, W_fc2), (update_gbf2, b_fc2)]);
    train_step = opt.apply_gradients([(update_gw_Wf1+tf.random_normal(shape=tf.shape(update_gw_Wf1), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/lot_size, W_fc1),
                                      (update_gbf1+tf.random_normal(shape=tf.shape(update_gbf1), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/lot_size, b_fc1),
                                      (update_gw_Wf2+tf.random_normal(shape=tf.shape(update_gw_Wf2), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/lot_size, W_fc2),
                                      (update_gbf2+tf.random_normal(shape=tf.shape(update_gbf2), mean=0.0, stddev=sigma * sensitivity,dtype=tf.float32)/lot_size, b_fc2)]);
    # train_step = opt.apply_gradients([(update_gw_Wf1+tf.random_normal(shape=tf.shape(update_gw_Wf1), mean=0.0, stddev=sigma * 1,dtype=tf.float32)/lot_size, W_fc1),
    #                                   (update_gbf1+tf.random_normal(shape=tf.shape(update_gbf1), mean=0.0, stddev=sigma * 1,dtype=tf.float32)/lot_size, b_fc1),
    #                                   (update_gw_Wf2+tf.random_normal(shape=tf.shape(update_gw_Wf2), mean=0.0, stddev=sigma * 1,dtype=tf.float32)/lot_size, W_fc2),
    #                                   (update_gbf2+tf.random_normal(shape=tf.shape(update_gbf2), mean=0.0, stddev=sigma * 1,dtype=tf.float32)/lot_size, b_fc2)]);
    # train_step = opt.apply_gradients([(update_gw_Wf1, W_fc1),
    #                                   (update_gbf1, b_fc1),
    #                                   (update_gw_Wf2, W_fc2),
    #                                   (update_gbf2, b_fc2)]);
    # train_step = opt.apply_gradients([(gw_W1, W_conv1), (gb1, b_conv1),(gw_W2,W_conv2),(gb2,b_conv2), (gw_Wf1, W_fc1), (gbf1, b_fc1),(gw_Wf1_2, W_fc1_2), (gbf1_2, b_fc1_2), (gw_Wf2, W_fc2), (gbf2, b_fc2)]);
    # 3层
    # train_step = opt.apply_gradients(
    #   [(gw_W1, W_conv1), (gb1, b_conv1), (gw_W2, W_conv2), (gb2, b_conv2),(gw_W3, W_conv3), (gb3, b_conv3),  (gw_Wf1, W_fc1), (gbf1, b_fc1),
    #    (gw_Wf2, W_fc2), (gbf2, b_fc2)]);

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1));
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

    start_time = time.time();
    max_test_accuracy=0

    sess.run(tf.global_variables_initializer())  # 全局变量初始值设定项

    # # 先实例化一个Saver()类
    # program = []
    # program += [W_conv1, b_conv1, W_conv2, b_conv2]
    # # program += [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
    # saver = tf.train.Saver(program)
    # saver.restore(sess, 'save_net_conv_wb.ckpt')
    j=0
    workbook = xlsxwriter.Workbook('/home/yxx/PythonProjects/PrivateDeepLearning-master/big_paper/mnist_sigma2_unchange_clipbound.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'epoch')
    worksheet.write(0, 1, 'train_accuracy')
    worksheet.write(0, 2, 'test_accuracy')

    for i in range(1,num_steps+1):
        # update_flag = 0
        # if(i % 6000 == 0):
        #     update_flag = 1
        # else:
        #     update_flag = 0
        if(i<=10*6000):
            now_learning_rate_by_epoch=0.1-0.0048*i/6000
        else:
            now_learning_rate_by_epoch = 0.052

        batch = mnist.train.next_batch(batch_size);

        if(i%60==1):
            # now_gw_W1 = sess.run(gw_W1,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gb1 = sess.run(gb1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gw_W2 = sess.run(gw_W2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gb2 = sess.run(gb2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gw_W3 = sess.run(gw_W3, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gb3 = sess.run(gb3, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            now_gw_Wf1,now_gbf1,now_gw_Wf2,now_gbf2 = sess.run([gw_Wf1,gbf1,gw_Wf2,gbf2], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

            # now_gw_Wf1 = sess.run(gw_Wf1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gbf1 = sess.run(gbf1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # # now_gw_Wf1_2 = sess.run(gw_Wf1_2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # # now_gbf1_2 = sess.run(gbf1_2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gw_Wf2 = sess.run(gw_Wf2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gbf2 = sess.run(gbf2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        else:
            # now_gw_W1  = now_gw_W1+sess.run(gw_W1,feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gb1 = now_gb1+sess.run(gb1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gw_W2 = now_gw_W2+sess.run(gw_W2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gb2 = now_gb2+sess.run(gb2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gw_W3 = now_gw_W3+sess.run(gw_W3, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gb3 = now_gb3+sess.run(gb3, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            now_gw_Wf1_i, now_gbf1_i, now_gw_Wf2_i, now_gbf2_i = sess.run([gw_Wf1, gbf1, gw_Wf2, gbf2],feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

            now_gw_Wf1 = now_gw_Wf1 + now_gw_Wf1_i
            now_gbf1 = now_gbf1 + now_gbf1_i
            # now_gw_Wf1_2 = now_gw_Wf1_2+now_gw_Wf1_2_i
            # now_gbf1_2 = now_gbf1_2+now_gbf1_2_i
            now_gw_Wf2 = now_gw_Wf2 + now_gw_Wf2_i
            now_gbf2 = now_gbf2 + now_gbf2_i

            # now_gw_Wf1 = now_gw_Wf1+sess.run(gw_Wf1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gbf1 = now_gbf1+sess.run(gbf1, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # # now_gw_Wf1_2 = now_gw_Wf1_2+sess.run(gw_Wf1_2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # # now_gbf1_2 = now_gbf1_2+sess.run(gbf1_2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gw_Wf2 = now_gw_Wf2+sess.run(gw_Wf2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # now_gbf2 = now_gbf2+sess.run(gbf2, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

            if (i % 60 == 0):
                # print(batch[0])
                # print(batch[1])

                # now_gw_W1 = now_gw_W1 / lot_size
                # now_gb1 = now_gb1 / lot_size
                # now_gw_W2 = now_gw_W2 / lot_size
                # now_gb2 = now_gb2 / lot_size
                # now_gw_W3 = now_gw_W3 / lot_size
                # now_gb3 = now_gb3 / lot_size
                now_gw_Wf1 = now_gw_Wf1 / lot_size
                now_gbf1 = now_gbf1 / lot_size
                # now_gw_Wf1_2 = now_gw_Wf1_2 / lot_size
                # now_gbf1_2 = now_gbf1_2 / lot_size
                now_gw_Wf2 = now_gw_Wf2 / lot_size
                now_gbf2 = now_gbf2 / lot_size

                train_step.run(feed_dict={ now_learning_rate:now_learning_rate_by_epoch, update_gw_Wf1: now_gw_Wf1,update_gbf1: now_gbf1,  update_gw_Wf2: now_gw_Wf2, update_gbf2: now_gbf2});
                # train_step.run(feed_dict={update_gw_W1:now_gw_W1, update_gb1:now_gb1, update_gw_W2:now_gw_W2, update_gb2:now_gb2, update_gw_Wf1:now_gw_Wf1,
                #                           update_gbf1:now_gbf1, update_gw_Wf1_2:now_gw_Wf1_2, update_gbf1_2:now_gbf1_2, update_gw_Wf2:now_gw_Wf2, update_gbf2:now_gbf2});

        if i % 6000 == 0:
            # train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0});
            # print("step \t %d \t training accuracy \t %g"%(i, train_accuracy));
            test_accuracy=accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            train_accuracy = accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0})
            print("step \t %d \t test accuracy \t %g \t train accuracy \t %g\t now_learning_rate=%g" % (i/60, test_accuracy,train_accuracy,now_learning_rate_by_epoch));
            j=j+1
            worksheet.write(j, 0, j)
            worksheet.write(j, 1, train_accuracy)
            worksheet.write(j, 2, test_accuracy)

            epsilon = comp_eps(32, sample_rate, sigma, i/60, delta)  # 使用给定sigma的abadi代码计算epsilon
            print("epsilon: {}".format(epsilon))
            if(i==6000):
                max_test_accuracy = test_accuracy
            elif (test_accuracy > max_test_accuracy):
                max_test_accuracy = test_accuracy
                max_test_accuracy_epsilon =epsilon
                max_test_accuracy_step=i/60
        sess.run([privacy_accum_op])
        spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
        # print(i, spent_eps_deltas)
        _break = False;
        for _eps, _delta in spent_eps_deltas:
            if _delta >= delta:
                _break = True;
                break;
        if _break == True:
            break;
    duration = time.time() - start_time;
    workbook.close()
    # save_path = saver.save(sess, "save_net_conv_wb.ckpt")
    # print("[+] Model saved in file: %s" % save_path)
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}));
    print("train accuracy %g" % accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0}));
    print(float(duration));
    print("max_test_accuracy:%g, max_test_accuracy_epsilon:%g, max_test_accuracy_step:%d" % (max_test_accuracy,max_test_accuracy_epsilon,max_test_accuracy_step));
    ###


if __name__ == '__main__':
    if tf.gfile.Exists('/tmp/mnist_logs'):
        tf.gfile.DeleteRecursively('/tmp/mnist_logs');
    tf.gfile.MakeDirs('/tmp/mnist_logs');

    parser = argparse.ArgumentParser();
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data');
    FLAGS = parser.parse_args();
    tf.app.run();