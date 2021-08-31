from __future__ import division
from __future__ import print_function

import sys
import os
#sys.path 当前py文件所在目录
#os.path.realpath(__file__) 获取当前执行脚本的绝对路径  [/Users/shiyingdan/GitHub/NPHard/train.py]
#os.path.dirname(path) 功能：去掉文件名，返回目录  [/Users/shiyingdan/GitHub/NPHard]
print('sys.path1:', sys.path)

sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )
print('sys.path2:', sys.path)
print('os.path.realpath(__file__)', os.path.realpath(__file__))
print('os.path.dirname(os.path.realpath(__file__)) ', os.path.dirname(os.path.realpath(__file__)) )


import time
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from copy import deepcopy

import tensorflow as tf
from utils import *
from models import GCN_DEEP_DIVER

N_bd = 32  # 输入的维度/特征的维度


# DEFINE_float(flag_name, default_value, docstring)
# tf.app.flags tf的命令行参数
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense' 模型？？？？？？？？？？？？？？
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')  # lr
flags.DEFINE_integer('epochs', 201, 'Number of epochs to train.')  # epoch
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')  # 隐藏层 1 的单元数量
flags.DEFINE_integer('diver_num', 32, 'Number of outputs.')  # 输出数量
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probaNUmbility).')  # dropout 比率
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')  # L2 损失（均方误差）的权重，L2损失又称均方误差、MSE？？？？？？？
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')  # 提前结束算法？？？？？？？？？？？？
flags.DEFINE_integer('max_degree', 1, 'Maximum Chebyshev polynomial degree.')  # 最大切比雪夫多项式度？？？？？？？？？？？？？
flags.DEFINE_integer('num_layer', 20, 'number of layers.')  # 层数，默认20层！

# Load data
data_path = "./data/CBS_Graph"
train_mat_names = os.listdir(data_path)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
print('train_mat_names', train_mat_names)

# Some preprocessing 预处理

num_supports = 1 + FLAGS.max_degree  # ？？？？？？？？？？？？？？？
model_func = GCN_DEEP_DIVER

# Define placeholders
# tf.sparse_placeholder 为稀疏张量插入占位符，该稀疏张量将始终被提供。
# tf.placeholder 是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
# 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, N_bd)), # featureless: #points
    'labels': tf.placeholder(tf.float32, shape=(None, 2)), # 0: not linked, 1:linked
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=N_bd, logging=True)

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES']=str(0)

# Initialize session
config = tf.ConfigProto()  # tf.ConfigProto()主要的作用是配置tf.Session的运算方式，比如gpu运算或者cpu运算
config.gpu_options.allow_growth = True  # gpu_options限制GPU资源使用，config.gpu_options.allow_growth = True动态申请显存
sess = tf.Session(config=config)
# 方法二：Initialize session
# gpu_options=tf.GPUOptions(allow_growth = True)
# config=tf.ConfigProto(gpu_options=gpu_options)
# session = tf.Session(config=config)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.outputs_softmax], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]

# Init variables
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state("result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

# cost_val = []

all_loss = np.zeros(2000, dtype=float)
all_acc = np.zeros(2000, dtype=float)

# Train model
for epoch in range(FLAGS.epochs):
    if os.path.isdir("result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32/%04d"%epoch):
        # os.path.isdir(),用于判断某一对象(需提供绝对路径)是否为目录
        # os.path.isfile(),用于判断某一对象(需提供绝对路径)是否为文件
        # os.listdir(),返回一个列表，其中包含有指定路径下的目录和文件的名称
        continue
    ct = 0  # 该参数ct是啥？？？？？？？？？？？
    os.makedirs("result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32/%04d" % epoch)
    # for id in np.random.permutation(len(train_mat_names)):
    for idd in range(2000):  # idd是啥变量未知？？？？？？？？？？？
        id = np.random.randint(338000)  # id 是啥意思也未知？？？？？？？？？？？原38000
        ct = ct + 1
        t = time.time()
        # load data
        mat_contents = sio.loadmat(data_path+'/'+train_mat_names[id])
        adj = mat_contents['adj']
        yy = mat_contents['indset_label']
        nn, nr = yy.shape  # number of nodes & results
        # y_train = yy[:,np.random.randint(0,nr)]
        # y_train = np.concatenate([1-np.expand_dims(y_train,axis=1), np.expand_dims(y_train,axis=1)],axis=1)

        # sample an intermediate graph
        yyr = yy[:,np.random.randint(0,nr)]
        yyr_num = np.sum(yyr)
        yyr_down_num = np.random.randint(0,yyr_num)
        if yyr_down_num > 0:
            yyr_down_prob = yyr * np.random.random_sample(yyr.shape)
            yyr_down_flag = (yyr_down_prob >= np.partition(yyr_down_prob,-yyr_down_num)[-yyr_down_num])
            tmp = np.sum(adj[yyr_down_flag, :], axis=0) > 0
            tmp = np.asarray(tmp).reshape(-1)
            yyr_down_flag[tmp] = 1
            adj_down = adj[yyr_down_flag==0,:]
            adj_down = adj_down[:,yyr_down_flag==0]
            yyr_down = yyr[yyr_down_flag==0]
            adj = adj_down
            nn = yyr_down.shape[0]
            yyr = yyr_down

        y_train = np.concatenate([1 - np.expand_dims(yyr, axis=1), np.expand_dims(yyr, axis=1)], axis=1)

        features = np.ones([nn, N_bd])
        features = sp.lil_matrix(features)
        features = preprocess_features(features)
        support = simple_polynomials(adj, FLAGS.max_degree)

        train_mask = np.ones([nn,1], dtype=bool)

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs], feed_dict=feed_dict)
        all_loss[ct-1] = outs[1]
        all_acc[ct-1] = outs[2]

        # Print results
        print('%03d %04d' % (epoch + 1, ct), "train_loss=", "{:.5f}".format(np.mean(all_loss[np.where(all_loss)])),
              "train_acc=", "{:.5f}".format(np.mean(all_acc[np.where(all_acc)])), "time=", "{:.5f}".format(time.time() - t))


    target=open("result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32/%04d/score.txt"%epoch,'w')
    target.write("%f\n%f\n"%(np.mean(all_loss[np.where(all_loss)]),np.mean(all_acc[np.where(all_acc)])))
    target.close()

    saver.save(sess,"result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32/model.ckpt")
    saver.save(sess,"result_IS4SAT_deep_ld32_c32_l20_cheb1_diver32_res32/%04d/model.ckpt"%epoch)

print("Optimization Finished!")