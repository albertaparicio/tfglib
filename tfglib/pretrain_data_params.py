# Created by Albert Aparicio on 03/02/2017
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os
from time import time

import numpy as np
from h5py import File as h5_File

from tfglib.construct_table import parse_file
from tfglib.utils import display_time


def pretrain_save_data_parameters(data_dir):
    # Save processing start time
    start_time = time()

    print('Starting')

    longest_sequence = 0
    files_list = []

    num_spk = len([entry for entry in os.scandir(data_dir) if entry.is_dir()])

    spk_max = np.zeros((num_spk, 42))
    spk_min = 1e+50 * np.ones((num_spk, 42))

    print("Processing speakers' data")
    for root, dirs, _ in os.walk(data_dir):
        for spk_index, a_dir in enumerate(dirs):
            for sub_root, _, sub_files in os.walk(os.path.join(root, a_dir)):
                # Get basenames of files in directory
                basenames = list(
                    set([file.split('.')[0] for file in sub_files]))

                for basename in basenames:
                    print('Processing ' + a_dir + ' -> ' + basename)

                    lf0_params = parse_file(
                        1,
                        os.path.join(sub_root, basename + '.lf0_log')
                    )

                    if lf0_params.shape[0] > longest_sequence:
                        longest_sequence = lf0_params.shape[0]

                    mcp_params = parse_file(
                        40,
                        os.path.join(sub_root, basename + '.cc')
                    )

                    mvf_params = parse_file(
                        1,
                        os.path.join(sub_root, basename + '.i.fv')
                    )

                    seq_params = np.concatenate(
                        (
                            mcp_params,
                            lf0_params,
                            mvf_params
                        ),
                        axis=1
                    )

                    # Compute maximum and minimum values
                    spk_max[spk_index, :] = np.maximum(
                        spk_max[spk_index, :], np.ma.max(seq_params, axis=0)
                    )
                    spk_min[spk_index, :] = np.minimum(
                        spk_min[spk_index, :], np.ma.min(seq_params, axis=0)
                    )

    print('Saving data to .h5 file')
    with h5_File(os.path.join(data_dir, 'pretrain_params.h5'), 'w') as f:
        # Save longest_sequence and max and min values as attributes
        f.attrs.create('longest_sequence', longest_sequence, dtype=int)
        f.attrs.create('speakers_max', spk_max)
        f.attrs.create('speakers_min', spk_min)

        f.close()

    print('Elapsed time: ' + display_time(time() - start_time))

    return longest_sequence, spk_max, spk_min


def pretrain_load_data_parameters(data_dir):
    # Load data from .h5 file
    with h5_File(os.path.join(data_dir, 'pretrain_params.h5'), 'r') as file:
        longest_sequence = file.attrs.get('longest_sequence')
        spk_max = file.attrs.get('speakers_max')
        spk_min = file.attrs.get('speakers_min')

        file.close()

    return longest_sequence, spk_max, spk_min
