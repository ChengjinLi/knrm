#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2017年07月06日

@author: lichengjin
"""
import numpy as np
import tensorflow as tf


def print_variable_info(var):
    """
    print variable info
    """
    print (var.op.name, ' ', var.get_shape().as_list())


class KNRMModel(object):
    """
    K-NRM：a kernel based neural ranking model
    """
    def __init__(self, input_query, input_doc, config, scope):
        with tf.name_scope('input'):
            print_variable_info(input_query)
            print_variable_info(input_doc)
            # num_per_entry在train过程中要设置成2，test过程大于等于1即可，方便一次处理 一个query与多个doc的score
            num_per_entry = input_doc.get_shape()[1].value
            self.learning_rate = config.learning_rate
            self.max_query_term_length = config.max_query_term_length
            self.max_doc_term_length = config.max_doc_term_length
            self.kernel_num = config.kernel_num
            self.init_scale = config.init_scale
            self.vocabulary_size = config.vocabulary_size
            self.embedding_dim = config.embedding_dim
            if 'relu' == config.activation:
                self.activation = tf.nn.relu
            else:
                self.activation = tf.nn.tanh
            if 'adam' == config.optimizer:
                self.optimizer = tf.train.AdamOptimizer
            else:
                self.optimizer = tf.train.GradientDescentOptimizer

        with tf.name_scope('embedding'):
            # look up embeddings for each term.
            self.embedding_weight = tf.get_variable(
                'embedding_weight',
                shape=[self.vocabulary_size + 1, self.embedding_dim],
                initializer=tf.random_uniform_initializer(-1 * self.init_scale, 1 * self.init_scale))
            # query_embedded, [batch_size, max_query_term_length, embedding_dim]
            self.query_embedded = tf.nn.embedding_lookup(self.embedding_weight, input_query, name='query_embedded')
            print_variable_info(self.query_embedded)
            # doc_embedded, [batch_size, num_per_entry, max_doc_term_length, embedding_dim]
            self.doc_embedded = tf.nn.embedding_lookup(self.embedding_weight, input_doc, name='doc_embedded')
            print_variable_info(self.doc_embedded)

        with tf.name_scope('translation'):
            # normalize and compute translation matrix.
            norm_query = tf.sqrt(tf.reduce_sum(tf.square(self.query_embedded), 2, keep_dims=True))
            self.query_embedded_normalized = self.query_embedded / norm_query
            norm_doc = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embedded), 3, keep_dims=True))
            doc_embedded_normalized = self.doc_embedded / norm_doc
            self.doc_embedded_normalized = tf.reshape(doc_embedded_normalized,
                                                      [-1, num_per_entry * self.max_doc_term_length,
                                                       self.embedding_dim])
            self.translation_matrix = tf.matmul(self.query_embedded_normalized,
                                                self.doc_embedded_normalized,
                                                transpose_b=True, name='translation_matrix')

        with tf.name_scope("kernel_pooling"):
            # Get the mu for each gaussian kernel
            self.mu_list = self.kernel_mu(self.kernel_num)
            # self.mu_list = tf.reshape(self.mu_list, shape=[1, 1, self.kernel_num])
            self.lamb = config.lamb
            # Get the sigma for each gaussian kernel
            self.sigma_list = self.kernel_sigma(self.kernel_num, self.lamb)
            # self.sigma_list = tf.reshape(self.sigma_list, shape=[1, 1, self.kernel_num])
            # compute Gaussian scores of each kernel
            tmp = tf.exp(-tf.square(tf.subtract(tf.expand_dims(self.translation_matrix, -1), self.mu_list)) / (2 * tf.square(self.sigma_list)))
            tmp_reshape = tf.reshape(tmp, [-1, num_per_entry, self.max_query_term_length, self.max_doc_term_length, self.kernel_num])
            # sum up gaussian scores
            kde = tf.reduce_sum(tmp_reshape, [3])
            # aggregated query terms，store the soft-TF features from each field.
            soft_tf_feats = tf.reduce_sum(tf.log(tf.maximum(kde, 1e-10)) * 0.01, [2])  # 0.01 used to scale down the data.
            # [batch, num_per_entry, n_bins]
            print "batch feature shape:", soft_tf_feats.get_shape()
            feats_flat = tf.reshape(soft_tf_feats, [-1, self.kernel_num])

        # Learning-To-Rank layer.
        with tf.name_scope("learning_to_rank"):
            tmp = np.sqrt(6.0 / (self.kernel_num + 1))
            self.weight = tf.get_variable(
                'weight',
                shape=[self.kernel_num, 1],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-1 * tmp, tmp)
            )
            self.bias = tf.get_variable(
                'bias',
                shape=[1],
                dtype=tf.float32,
                initializer=tf.constant_initializer(1.0)
            )

        with tf.name_scope("output"):
            # scores is the final matching score.
            scores = self.activation(tf.matmul(feats_flat, self.weight) + self.bias)
            self.scores = tf.reshape(scores, [-1, num_per_entry])
            print "scores: ", self.scores
            self.pos_scores = tf.slice(self.scores, [0, 0], [-1, 1], name='pos_scores')
            print "pos_scores: ", self.pos_scores
            self.neg_scores = tf.slice(self.scores, [0, 1], [-1, -1], name='neg_scores')
            print "neg_scores: ", self.neg_scores
            # loss, max(0, 1 - score1 + score2)
            self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.pos_scores + tf.reduce_mean(self.neg_scores, 1)))
            tf.summary.scalar("loss", self.loss)

        # self.global_step = tf.train.get_or_create_global_step()
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        optimizer = self.optimizer(self.learning_rate)
        self.grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

        grad_summaries = []
        for grad, var in self.grads_and_vars:
            if grad is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(var.name), grad)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(var.name), tf.nn.zero_fraction(grad))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))

    @staticmethod
    def kernel_mu(kernel_num):
        """
        计算每个高斯核的均值, 均值设置为每个bin的中值
        Get the mu for each gaussian kernel，mu is the middle of each bin
        :param kernel_num: the number of kernels including exact match，first one is exact match
        :return: mu_list, a list of mu. e.g.：[1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        """
        mu_list = [1.0]  # for exact match
        if kernel_num == 1:
            return mu_list
        bin_size = 2.0 / (kernel_num - 1)  # score range from [-1, 1]
        mu_list.append(1.0 - bin_size / 2.0)  # mu: middle of the bin
        for i in xrange(1, kernel_num - 1):
            mu_list.append(mu_list[i] - bin_size)
        print "kernel mu values: ", mu_list
        return mu_list

    @staticmethod
    def kernel_sigma(kernel_num, lamb):
        """
        计算每个高斯核的标准差
        :param kernel_num: the number of kernels including exact match，first one is exact match
        :param lamb: use to the gaussian kernels sigma value, sigma = lamb * bin_size
        :return: sigma_list, a list of simga. e.g.：[1e-5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        """
        sigma_list = [0.00001]  # for exact match. small variance -> exact match
        if kernel_num == 1:
            return sigma_list
        bin_size = 2.0 / (kernel_num - 1)  # score range from [-1, 1]
        sigma_list += [bin_size * lamb] * (kernel_num - 1)
        print "kernel sigma values: ", sigma_list
        return sigma_list
