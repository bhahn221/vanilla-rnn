# replication of https://corochann.com/penn-tree-bank-ptb-dataset-introduction-1456.html

from __future__ import print_function
import os
import matplotlib.pyplot as plt

import numpy as np

import chainer
import itertools

# loading dictionary
ptb_dict = chainer.datasets.get_ptb_words_vocabulary()

print('Number of vacabulary', len(ptb_dict))

ptb_word_id_dict = ptb_dict
ptb_id_word_dict = dict((v, k) for k, v in ptb_word_id_dict.items())
ptb_id_word_dict[10000] = '<bos>'

def split_by_delimiter(dataset_list, delimiter, start):
    split_list = [list(y) for x, y in itertools.groupby(dataset_list, lambda z: z==delimiter) if not x]
    
    for split in split_list:
        split.append(delimiter)
        split.insert(0, start)

    return split_list

def index_to_word(index_list):
    word_list = ' '.join([ptb_id_word_dict[i] for i in index_list])

    return word_list

def load_dataset():
    # loading dataset
    train, val, test = chainer.datasets.get_ptb_words()

    print('train type:', type(train), train.shape, train)
    print('val   type:', type(val), val.shape, val)
    print('test  type:', type(test), test.shape, test)

    train_set = split_by_delimiter(train, 24, 10000)
    val_set = split_by_delimiter(val, 24, 10000)
    test_set = split_by_delimiter(test, 24, 10000)

    return train_set, val_set, test_set
