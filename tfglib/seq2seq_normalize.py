# Created by Albert Aparicio on 4/12/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np


def maxmin_scaling(src_data_matrix, src_data_mask, trg_data_matrix, trg_data_mask, max_mat, min_mat):
    # Mask data
    src_masked_data = np.ma.array(src_data_matrix,
                                  mask=1 - np.repeat(np.reshape(src_data_mask, (src_data_matrix.shape[0], 1)),
                                                     src_data_matrix.shape[1], axis=1))
    trg_masked_data = np.ma.array(trg_data_matrix,
                                  mask=1 - np.repeat(np.reshape(trg_data_mask, (trg_data_matrix.shape[0], 1)),
                                                     trg_data_matrix.shape[1], axis=1))

    # Compute speaker indexes
    src_spk_index = np.argmax(np.ma.argmax(src_masked_data[:, 44:54], axis=0, fill_value=0))
    trg_spk_index = np.argmax(np.ma.argmax(src_masked_data[:, 54:64], axis=0, fill_value=0))

    # Obtain maximum and minimum values matrices
    src_spk_max = max_mat[src_spk_index, :]
    src_spk_min = min_mat[src_spk_index, :]

    trg_spk_max = max_mat[trg_spk_index, :]
    trg_spk_min = min_mat[trg_spk_index, :]

    # Compute minmax scaling
    src_norm = (src_masked_data[:, 0:42] - src_spk_min) / (src_spk_max - src_spk_min)
    trg_norm = (trg_masked_data[:, 0:42] - trg_spk_min) / (trg_spk_max - trg_spk_min)

    return (src_norm, trg_norm)
