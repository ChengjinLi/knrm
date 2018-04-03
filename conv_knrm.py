#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2018年03月23日

@author: lichengjin
"""
import tensorflow as tf


def print_variable_info(var):
    """
    print variable info
    """
    print (var.op.name, ' ', var.get_shape().as_list())


class ConvKNRMModel(object):
    """
    K-NRM：a kernel based neural ranking model
    """
    def __init__(self, input_query, input_doc, config, scope):
        with tf.name_scope('input'):
            self.input_query = input_query
            self.input_doc = input_doc
            print_variable_info(input_query)
            print_variable_info(input_doc)
            # num_per_entry在train过程中要设置成2，test过程大于等于1即可，方便一次处理 一个query与多个doc的score
            num_per_entry = input_doc.get_shape()[1].value
            self.learning_rate = config.learning_rate
            self.max_query_term_length = config.max_query_term_length
            self.max_doc_term_length = config.max_doc_term_length
            self.num_filters = config.num_filters
            self.kernel_num = config.kernel_num
            self.init_scale = config.init_scale
            self.vocabulary_size = config.vocabulary_size
            self.embedding_dim = config.embedding_dim
            self.cross = config.cross
            self.max_ngram = config.max_ngram
            self.use_exact = config.use_exact
            # Model parameters for feedfoward rank NN
            if self.cross:  # 不同粒度交叉匹配，所以是self.max_ngram * self.max_ngram
                self.total_bins = self.kernel_num * self.max_ngram * self.max_ngram
            else:  # 相同粒度匹配，所以是self.max_ngram
                self.total_bins = self.kernel_num * self.max_ngram
            if 'relu' == config.activation:
                self.activation = tf.nn.relu
            else:
                self.activation = tf.nn.tanh
            if 'adam' == config.optimizer:
                self.optimizer = tf.train.AdamOptimizer
            else:
                self.optimizer = tf.train.GradientDescentOptimizer
            # Get the mu for each gaussian kernel
            self.mu_list = self.kernel_mu(self.kernel_num, self.use_exact)
            self.mu_list = tf.reshape(self.mu_list, shape=[1, 1, self.kernel_num])
            self.lamb = config.lamb
            # Get the sigma for each gaussian kernel
            self.sigma_list = self.kernel_sigma(self.kernel_num, self.lamb)
            self.sigma_list = tf.reshape(self.sigma_list, shape=[1, 1, self.kernel_num])

        with tf.name_scope('embedding'):
            # look up embeddings for each term.
            self.embedding_weight = tf.get_variable(
                'embedding_weight',
                shape=[self.vocabulary_size + 1, self.embedding_dim],
                initializer=tf.random_uniform_initializer(-1 * self.init_scale, 1 * self.init_scale))
            # query_embedded, [batch_size, max_query_term_length, embedding_dim]
            self.query_embedded = tf.nn.embedding_lookup(self.embedding_weight, self.input_query, name='query_embedded')
            print_variable_info(self.query_embedded)
            # query_embedded_expanded, [batch_size, max_query_term_length, embedding_dim, 1]
            self.query_embedded_expanded = tf.expand_dims(self.query_embedded, -1)
            print_variable_info(self.query_embedded_expanded)
            # input_doc_rs, [batch_size * num_per_entry, max_doc_term_length]
            self.input_doc_rs = tf.reshape(self.input_doc, [-1, self.max_doc_term_length])
            # doc_embedded, [batch_size * num_per_entry, max_doc_term_length, embedding_dim]
            self.doc_embedded = tf.nn.embedding_lookup(self.embedding_weight, self.input_doc_rs, name='doc_embedded')
            print_variable_info(self.doc_embedded)
            # doc_embedded_expanded, [batch_size * num_per_entry, max_doc_term_length, embedding_dim, 1]
            self.doc_embedded_expanded = tf.expand_dims(self.doc_embedded, -1)
            print_variable_info(self.doc_embedded_expanded)

        # Model parameters for convolutions
        query_embedded_list = []
        doc_embedded_list = []
        for h in range(1, self.max_ngram + 1):
            with tf.variable_scope("conv-{0}-gram".format(h)):
                # 卷积层
                filter_shape = [h, self.embedding_dim, 1, self.num_filters]
                # weight是卷积的输入矩阵
                # 利用truncated_normal生成截断正态分布随机数, 尺寸是filter_shape, 均值mean, 标准差stddev,
                # 不过只保留[mean-2*stddev, mean+2*stddev]范围内的随机数
                weight = tf.get_variable(
                    name='conv{0}_weight'.format(h),
                    shape=filter_shape,
                    dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                # bias是卷积的输入偏置量
                bias = tf.get_variable(
                    name='conv{0}_bias'.format(h),
                    shape=[self.num_filters],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer(0.1))
                # 卷积操作, “VALID”表示使用narrow卷积
                # query_conv， [batch_size, max_query_term_length - h + 1, 1, num_filters]
                query_conv = tf.nn.conv2d(
                    input=self.query_embedded_expanded,
                    filter=weight,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name='conv{0}_query'.format(h))
                print_variable_info(query_conv)
                # doc_conv， [batch_size * num_per_entry, max_doc_term_length - h + 1, 1, num_filters]
                doc_conv = tf.nn.conv2d(
                    input=self.doc_embedded_expanded,
                    filter=weight,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name='conv{0}_doc'.format(h))
                print_variable_info(doc_conv)

                query_h = tf.nn.relu(tf.nn.bias_add(query_conv, bias)) + 0.000000001
                doc_h = tf.nn.relu(tf.nn.bias_add(doc_conv, bias)) + 0.000000001

                query_h = tf.squeeze(query_h)
                doc_h = tf.squeeze(doc_h)

                query_embedded_normalized = tf.nn.l2_normalize(query_h, 2)
                doc_embedded_normalized = tf.nn.l2_normalize(doc_h, 2)
                print ("query_embedded_normalized")
                print_variable_info(query_embedded_normalized)
                print ("doc_embedded_normalized")
                print_variable_info(doc_embedded_normalized)

                query_embedded_list.append(query_embedded_normalized)
                doc_embedded_list.append(doc_embedded_normalized)

        translation_matrix_list = []
        kernel_pooling_outputs = []
        for h1_idx, query_emb in enumerate(query_embedded_list):
            for h2_idx, doc_emb in enumerate(doc_embedded_list):
                if h1_idx != h2_idx and not self.cross:
                    continue
                doc_emb = tf.reshape(doc_emb, [-1, num_per_entry * (self.max_doc_term_length - h2_idx), self.num_filters])
                # translation_matrix, [batch_size, max_query_term_length, num_per_entry * max_doc_term_length]
                translation_matrix = tf.matmul(query_emb, doc_emb, transpose_b=True, name='translation_matrix')
                translation_matrix_rs = tf.expand_dims(translation_matrix, -1)
                print_variable_info(translation_matrix_rs)
                translation_matrix_list.append(translation_matrix_rs)

                # kernel_pooling，compute Gaussian scores of each kernel
                tmp = tf.exp(-tf.square(tf.subtract(translation_matrix_rs, self.mu_list)) / 2 * tf.square(self.sigma_list))
                tmp_reshape = tf.reshape(tmp, [-1, num_per_entry, self.max_query_term_length - h1_idx, self.max_doc_term_length - h2_idx, self.kernel_num])
                # sum up gaussian scores
                kde = tf.reduce_sum(tmp_reshape, [3])
                # aggregated query terms，store the soft-TF features from each field.
                # soft_tf_feats, [batch_size, num_per_entry, kernel_num]
                soft_tf_feats = tf.reduce_sum(tf.log(tf.maximum(kde, 1e-10)) * 0.01, [2])  # 0.01 used to scale down the data.
                kernel_pooling_outputs.append(soft_tf_feats)
                print_variable_info(soft_tf_feats)
                # feats_flat = tf.reshape(soft_tf_feats, [-1, self.kernel_num])

        # Learning-To-Rank layer.
        with tf.name_scope("learning_to_rank"):
            # [batch_size, num_per_entry, total_bins]
            all_kernel_pooling_output = tf.concat(kernel_pooling_outputs, 2)
            feats_flat = tf.reshape(all_kernel_pooling_output, [-1, self.total_bins])

            print_variable_info(all_kernel_pooling_output)
            self.ltr_weight = tf.get_variable(
                'ltr_weight',
                shape=[self.total_bins, 1],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.ltr_bias = tf.get_variable(
                'ltr_bias',
                dtype=tf.float32,
                initializer=tf.zeros([1])
            )
            # scores is the final matching score.
            scores = self.activation(tf.matmul(feats_flat, self.ltr_weight) + self.ltr_bias)
            self.scores = tf.reshape(scores, [-1, num_per_entry])
            print_variable_info(self.scores)
            # hinge loss
            # self.pos_scores = tf.slice(self.scores, [0, 0], [-1, 1], name='pos_scores')
            # print "pos_scores: ", self.pos_scores
            # self.neg_scores = tf.slice(self.scores, [0, 1], [-1, -1], name='neg_scores')
            # print "neg_scores: ", self.neg_scores
            # self.pos_scores = tf.tile(self.pos_scores, [1, tf.shape(self.neg_scores)[1]])
            # # loss, max(0, 1 - score1 + score2)
            # self.loss = tf.reduce_mean(tf.reduce_mean(
            #     tf.maximum(0.0, 1 - self.pos_scores + self.neg_scores), 1))
            # cross_entropy
            gamma = tf.get_variable("loss_gamma", initializer=1., trainable=True)
            self.scores = self.scores * gamma
            label = tf.zeros(tf.stack([tf.shape(self.scores)[0]]), dtype=tf.int32)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=label)
            self.loss = tf.reduce_mean(self.loss)
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
    def kernel_mu(kernel_num, use_exact):
        """
        计算每个高斯核的均值, 均值设置为每个bin的中值
        Get the mu for each gaussian kernel，mu is the middle of each bin
        :param kernel_num: the number of kernels including exact match，first one is exact match
        :param use_exact: use exact or not
        :return: mu_list, a list of mu. e.g.：[1, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9]
        """
        if use_exact:
            mu_list = [1.0]  # for exact match
        else:
            mu_list = [2.0]
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
