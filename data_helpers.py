#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年06月21日

@author: lichengjin
"""
from __future__ import absolute_import
import os
import math
import random
import argparse
import numpy as np
import tensorflow as tf

SEPARATOR = ','


def get_config_args():
    parser = argparse.ArgumentParser(description='knrm train argument')
    parser.add_argument('--train-data-path', type=str, default='/home/search/lichengjin/data/dssm_training_data/part-00001',
                        help='the path that keeps the train data file')
    parser.add_argument('--test-data-path', type=str, default='/home/search/lichengjin/data/test',
                        help='the path that keeps the test data file')
    # parser.add_argument('--train-data-path', type=str, default='./data/train.pairs.hashed.shuf',
    #                     help='the path that keeps the train data file')
    # parser.add_argument('--test-data-path', type=str, default='./data/dev.pairs.hashed.shuf',
    #                     help='the path that keeps the test data file')
    parser.add_argument('--model-dir', type=str, default='./model', help='the directory that keeps the model file')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--optimizer', type=str, default='grad', help='optimizer')
    parser.add_argument('--num-epochs', type=int, default=10, help='the number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='the batch size')
    parser.add_argument('--embedding-dim', type=int, default=200, help='Dimensionality of word embedding')
    parser.add_argument('--vocabulary-size', type=int, default=2000000, help='Size of vocabulary')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='the learning rate')
    parser.add_argument('--max-query-term-length', type=int, default=20, help='the max query term length')
    parser.add_argument('--max-doc-term-length', type=int, default=20, help='the max doc term length')
    parser.add_argument('--kernel-num', type=int, default=11,
                        help='the number of kernels, default: 11. One exact match kernel and 10 soft kernels')
    parser.add_argument('--lamb', type=float, default=0.5,
                        help='use to the gaussian kernels sigma value, sigma = lamb * bin_size')
    parser.add_argument('--init-scale', type=float, default=0.1, help='init scale')
    parser.add_argument('--show-freq', type=int, default=10, help='Show train results after this many steps')
    parser.add_argument('--test-freq', type=int, default=10, help='Test model results after this many steps')
    parser.add_argument('--save-freq', type=int, default=10, help='save model each number steps')
    parser.add_argument('--test-num', type=int, default=10000, help='Test sample size')

    config = parser.parse_args()
    return config


def batch_iter(data, batch_size):
    """
    Generates data batch iterator for data set.
    """
    data_size = len(data)
    num_batches_per_epoch = int(math.ceil(float(len(data))/batch_size))
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]


def load_all_data_at_once(data_path, max_query_term_length, max_doc_term_length, max_num=None):
    """
    Load train data，when the amount of data is small(less than 100G), select this method
    """
    if not os.path.exists(data_path):
        print ("%s is not exist!" % data_path)
        exit(0)
    else:
        print ("%s is exist!" % data_path)
    test_data = np.loadtxt(data_path, dtype=np.int32, delimiter=SEPARATOR)
    if max_num:
        # 设置随机数种子
        np.random.seed(10)
        # 返回一个随机后的的数组,并取前max_num个
        shuffle_indices = np.random.permutation(np.arange(len(test_data)))
        test_data = test_data[shuffle_indices][:max_num]

    test_data_split = np.split(test_data, [max_query_term_length, max_query_term_length + max_doc_term_length], axis=1)
    query_term_ids = test_data_split[0]
    pos_doc_term_ids = test_data_split[1]
    neg_doc_term_ids = test_data_split[2]
    doc_term_ids = np.append(pos_doc_term_ids[:, np.newaxis], neg_doc_term_ids[:, np.newaxis], 1)
    print("query_term_ids shape: {:s}".format(query_term_ids.shape))
    print("doc_term_ids shape: {:s}".format(doc_term_ids.shape))
    return query_term_ids, doc_term_ids


def load_batch_data_by_queue(filename_list, max_query_term_length, max_doc_term_length,
                             num_epochs=10, batch_size=256, num_threads=12):
    """
    Load batch data by queue，when the amount of data is big(more than 100G), select this method
    """
    for filename in filename_list:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs=num_epochs)
    reader = tf.TextLineReader()
    _, records = reader.read_up_to(filename_queue, batch_size)
    # record_defaults = [tf.constant([], dtype=tf.int32)] * (max_query_term_length + 2 * max_doc_term_length)
    record_defaults = [tf.constant([], dtype=tf.int32)] * (max_query_term_length + 5 * max_doc_term_length)
    temp_tensor = tf.stack(tf.decode_csv(records, record_defaults, SEPARATOR), 1)
    # split_size = [max_query_term_length] + [max_doc_term_length] * 2
    split_size = [max_query_term_length] + [max_doc_term_length * 5]
    # query_term_ids, pos_doc_term_ids, neg_doc_term_ids = tf.split(temp_tensor, split_size, 1)
    query_term_ids, doc_term_ids = tf.split(temp_tensor, split_size, 1)
    doc_term_ids = tf.reshape(doc_term_ids, [-1, 5, max_doc_term_length])
    # doc_term_ids = tf.concat([tf.expand_dims(pos_doc_term_ids, 1), tf.expand_dims(neg_doc_term_ids, 1)], axis=1)
    batch_data = tf.train.shuffle_batch((query_term_ids, doc_term_ids),
                                        batch_size=batch_size,
                                        capacity=batch_size*200,
                                        min_after_dequeue=batch_size*10,
                                        num_threads=num_threads,
                                        enqueue_many=True)

    return batch_data
