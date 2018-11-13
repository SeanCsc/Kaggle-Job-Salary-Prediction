# Kaggle-Job-Salary-Prediction

好久没有打过kaggle比赛了，之前所参加的kaggle或者课程项目都以单一数据类型为主，比如图像数据或者时间序列。

趁着最近在学cs224n-nlp，解锁带文本特征的任务入门。这个kaggle记录学习的过程。

链接： https://www.kaggle.com/c/job-salary-prediction/data

目标： 根据文本描述（unstructure text）和一些辅助特征（structured)预测该工作的薪水。

## 特征分析

text部分： title和description

categorical部分： "Category", "Company", "LocationNormalized", "ContractType", "ContractTime"

target: salary

## 预处理和特征工程

首先，观察到target并非正态分布，可能会有一些极端高的薪水，因此先做转化（log1p)

**文本部分** ：这里是本次项目的重点。

1. 从token和lowercase开始。

2. Counter()容器记录每个词出现的次数 - 过滤掉小于阈值的词

3. 建立token到id的字典

4. 根据自己建立的词库将句子转化为矩阵

**其他部分** 

1. 对categorical特征编码：one-hot / TF-idf

**数据分割**

1. train_test_split - 分割数据，并且记录下来其index

## 模型搭建
本次使用神经网络模型

1.切分mini-batches

2. 三个部分（title,description,others)分别训练一个encoder,然后喂给一个总的神经网络

## 模型解释

神经网络通常不像线性模型那样具有比较强的可解释性。所以以下是几个探索的方向：

1.看对输入的波动会产生什么样的变化

2. 线性拟合

3. 

## 学习到的问题：
1.预测的是log1p,如何返回最终的结果？

2.如何在每一轮训练后输出metrics

