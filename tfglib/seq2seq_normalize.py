# Created by Albert Aparicio on 4/12/16
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import numpy as np


def mask_data(data_matrix, data_mask):
  """Assumes it is input one sequence only"""
  return np.ma.array(
      data_matrix,
      mask=np.logical_not(np.repeat(
          np.reshape(data_mask, (data_matrix.shape[0], 1)),
          data_matrix.shape[1],
          axis=1
          ))
      )


def maxmin_scaling(
    src_matrix, src_mask, trg_matrix, trg_mask, src_max_mat, src_min_mat,
    trg_max_mat, trg_min_mat):
  # Mask data
  src_masked = mask_data(src_matrix, src_mask)
  trg_masked = mask_data(trg_matrix, trg_mask)

  # Obtain maximum and minimum values matrices
  src_spk_max = src_max_mat[int(src_matrix[0, 44]), :]
  src_spk_min = src_min_mat[int(src_matrix[0, 44]), :]

  trg_spk_max = trg_max_mat[int(src_matrix[0, 45]), :]
  trg_spk_min = trg_min_mat[int(src_matrix[0, 45]), :]

  # Compute minmax scaling
  src_norm = (src_masked[:, 0:42] - src_spk_min) / (src_spk_max - src_spk_min)
  trg_norm = (trg_masked[:, 0:42] - trg_spk_min) / (trg_spk_max - trg_spk_min)

  return src_norm, trg_norm


def unscale_prediction(src_matrix, src_mask, scaled_pred, max_mat, min_mat):
  src_masked_data = mask_data(src_matrix, src_mask)

  src_spk_index = int(src_masked_data[0, 44])

  src_spk_max = max_mat[src_spk_index, :]
  src_spk_min = min_mat[src_spk_index, :]

  return scaled_pred * (src_spk_max - src_spk_min) + src_spk_min
