# Created by Albert Aparicio on 27/11/2016
# coding: utf-8

import numpy as np


def zero_pad_params(max_length, mode, params_matrix):
    """Function to apply a zero-padding to a parameters matrix"""
    # Pythonic way to implement a switch-case clause
    # https://bytebaker.com/2008/11/03/switch-case-statement-in-python/
    options = {'src': src_zero_pad,
               'trg': trg_zero_pad}

    return options[mode](max_length, params_matrix)


def src_zero_pad(max_length, params_matrix):
    return np.concatenate((
        np.zeros((
            max_length - params_matrix.shape[0],
            params_matrix.shape[1]
        )),
        params_matrix
    ))


def trg_zero_pad(max_length, params_matrix):
    return np.concatenate((
        params_matrix,
        np.zeros((
            max_length - params_matrix.shape[0],
            params_matrix.shape[1]
        ))
    ))
