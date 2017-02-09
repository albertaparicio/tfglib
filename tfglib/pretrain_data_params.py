# Created by Albert Aparicio on 03/02/2017
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os
from sys import version_info
from time import time

import numpy as np
from h5py import special_dtype, File

from tfglib.construct_table import parse_file
from tfglib.utils import display_time, int2pair

if version_info.major == 2:
    import scandir


def replicate_frames(params_mat, max_seq_length, values, probabilities):
    # TODO Document this function
    src_res = np.zeros((int(max_seq_length), int(params_mat.shape[1])))
    trg_res = src_res.copy()

    src_i = trg_i = 0

    for i in range(params_mat.shape[0]):
        p = int2pair(np.random.choice(values, p=probabilities))

        frame = params_mat[i, :]

        src_res[src_i:src_i + p[0], :] = np.repeat(
            np.reshape(frame, (1, frame.shape[0])), p[0], axis=0
        )

        trg_res[trg_i:trg_i + p[1], :] = np.repeat(
            np.reshape(frame, (1, frame.shape[0])), p[1], axis=0
        )

        src_i += p[0]
        trg_i += p[1]

        src_mask = np.concatenate((
            np.ones((
                src_i,
                1
            )),
            np.zeros((
                max_seq_length - src_i,
                1
            ))
        ))

        trg_mask = np.concatenate((
            np.ones((
                trg_i,
                1
            )),
            np.zeros((
                max_seq_length - trg_i,
                1
            ))
        ))

    return src_res, trg_res, src_mask, trg_mask


def pretrain_save_data_parameters(
        data_dir,
        speakers_file='speakers.list',
        params_file='pretrain_params.h5',
):
    # TODO Document this function
    # Save processing start time
    start_time = time()

    print('Starting')

    longest_sequence = 0
    files_list = []

    num_spk = len([entry for entry in os.scandir(data_dir) if entry.is_dir()])

    spk_max = np.zeros((num_spk, 42))
    spk_min = 1e+50 * np.ones((num_spk, 42))

    speakers = open(os.path.join(data_dir, speakers_file), 'r').readlines()
    # Strip '\n' characters
    dirs = [line.split('\n')[0] for line in speakers]

    print("Processing speakers' data")
    for spk_index, a_dir in enumerate(dirs):
        for sub_root, _, sub_files in os.walk(os.path.join(data_dir, a_dir)):
            # Get basenames of files in directory
            basenames = list(set([os.path.join(
                sub_root,
                file.split('.')[0]
            ) for file in sub_files]))

            files_list += basenames

            for basename in basenames:
                print('Processing ' + basename)

                lf0_params = parse_file(
                    1,
                    basename + '.lf0_log'
                )

                if lf0_params.shape[0] > longest_sequence:
                    longest_sequence = lf0_params.shape[0]

                mcp_params = parse_file(
                    40,
                    basename + '.cc'
                )

                mvf_params = parse_file(
                    1,
                    basename + '.i.fv'
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

    with File(os.path.join(data_dir, params_file), 'w') as f:
        # Save longest_sequence and the max and min values as attributes
        f.attrs.create('longest_sequence', longest_sequence, dtype=int)
        f.attrs.create('speakers_max', spk_max)
        f.attrs.create('speakers_min', spk_min)

        # TODO Support Python 2
        # sys.version_info -> Get running Python version
        dt = special_dtype(vlen=str)

        utf_list = [n.encode(
            encoding="utf-8",
            errors="ignore"
        ) for n in files_list]
        f.create_dataset(
            name='files_list',
            shape=(len(utf_list), 1),
            data=utf_list,
            dtype=dt
        )

        f.close()

    print('Elapsed time: ' + display_time(time() - start_time))

    return int(np.floor(longest_sequence * 1.3)), spk_max, spk_min, files_list


def pretrain_load_data_parameters(data_dir, params_file='pretrain_params.h5'):
    # TODO Document this function
    # Load data from .h5 file
    with File(os.path.join(data_dir, params_file), 'r') as file:
        longest_sequence = file.attrs.get('longest_sequence')
        spk_max = file.attrs.get('speakers_max')
        spk_min = file.attrs.get('speakers_min')

        files_list_encoded = file['files_list'][:]
        files_list = [n[0] for n in files_list_encoded]

        file.close()

    # Increase the size of the maximum sequence length, to allow the
    # longest sequence's frames to be replicated when training
    return int(np.floor(longest_sequence * 1.3)), spk_max, spk_min, files_list


def pretrain_train_generator(
        params_path,
        dtw_prob_file,
        params_file='pretrain_params.h5',
        speakers_file='speakers.list',
        basename_len=11,
        validation=False,
        val_fraction=None
):
    # TODO Document this function
    speakers = open(os.path.join(params_path, speakers_file), 'r').readlines()
    # Strip '\n' characters
    speakers = [line.split('\n')[0] for line in speakers]

    # Read data from .h5 files
    (longest_sequence, spk_max, spk_min, files_list
     ) = pretrain_load_data_parameters(params_path, params_file)

    # Increase the size of the maximum sequence length, to allow the
    # longest sequence's frames to be replicated
    # longest_sequence = np.floor(longest_sequence * 1.3)

    # Take the files for training or for validation
    if validation:
        # Take the validation fraction from the end of the list
        files_list = files_list[int(np.floor(-val_fraction * len(files_list))):]
    else:
        # Take from the beginning of the list until the validation fraction
        files_list = files_list[:int(np.floor(-val_fraction * len(files_list)))]

    with File(os.path.join(params_path, dtw_prob_file), 'r') as f:
        # Save numbers and probabilities
        values = f['values'][:]
        probabilities = f['probabilities'][:]

        f.close()

    while True:
        # Iterate over files list
        for basename in files_list:
            # Compute speaker index
            spk_index = speakers.index(
                str(basename[-1 * (basename_len + 3):-1 * (basename_len + 1)])
            )

            # Read parameters
            mcp_params = parse_file(40, basename + '.cc')
            lf0_params = parse_file(1, basename + '.lf0_log')
            mvf_params = parse_file(1, basename + '.i.fv')
            uv_flags = parse_file(1, basename + '.lf0_log.uv_mask')

            # Get max and min values for each speaker
            src_spk_max = spk_max[spk_index, :]
            src_spk_min = spk_min[spk_index, :]

            # Maxmin normalize
            src_normalized = (np.concatenate((
                mcp_params,
                lf0_params,
                mvf_params), axis=1
            ) - src_spk_min) / (src_spk_max - src_spk_min)

            # One-hot encode the speaker indexes
            spk_index_vector = np.repeat(
                spk_index,
                lf0_params.shape[0],
                axis=0
            )

            # Construct End-Of-Sequence flags
            eos_flags = np.zeros(lf0_params.shape[0])
            eos_flags[-1] = 1

            # Construct sequence "slice"
            seq_params = np.concatenate((
                src_normalized,
                uv_flags,
                eos_flags,
                spk_index_vector,
                spk_index_vector
            ), axis=1)

            # Replicate frames with dtw probabilities
            (src_res, trg_res, src_mask, trg_mask) = replicate_frames(
                seq_params,
                longest_sequence,
                values,
                probabilities
            )

            # Flip frames and return them
            yield (np.fliplr(src_res), trg_res, src_mask)
