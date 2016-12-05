# Created by Albert Aparicio on 4/12/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function


def maxmin_scaling(data_matrix, max_mat, min_mat):
    return (data_matrix - min_mat) / (max_mat - min_mat)
