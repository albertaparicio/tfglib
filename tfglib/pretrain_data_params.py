# Created by Albert Aparicio on 03/02/2017
# coding: utf-8

# This import makes Python use 'print' as in Python 3.x
from __future__ import print_function

import os
from random import shuffle
from sys import version_info
from time import time

import numpy as np
from h5py import special_dtype, File

from tfglib.construct_table import parse_file
from tfglib.utils import display_time, int2pair

if version_info.major == 3 and version_info.minor >= 5:
    from os import scandir
else:
    from scandir import scandir


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

    # Target data must have 44 parameters
    return src_res, trg_res[:, 0:44], src_mask, trg_mask


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

    num_spk = len([entry for entry in scandir(data_dir) if entry.is_dir()])

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
    longest_sequence = int(np.floor(longest_sequence * 1.7))

    return longest_sequence, spk_max, spk_min, files_list


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
    longest_sequence = int(np.floor(longest_sequence * 1.7))

    return longest_sequence, spk_max, spk_min, files_list


def pretrain_train_generator(
        params_path,
        batch_size,
        params_file='pretrain_params.h5',
        validation=False,
        val_fraction=0.25,
        basename_len=11,
        replicate=True
):
    # TODO Document this function

    # Read data from .h5 files
    (longest_sequence, spk_max, spk_min, files_list
     ) = pretrain_load_data_parameters(params_path, params_file)

    # Take the files for training or for validation
    if validation:
        # Take the validation fraction from the end of the list
        files_list = files_list[
                     int(np.floor(-1 * val_fraction * len(files_list))):
                     ]
    else:
        # Take from the beginning of the list until the validation fraction
        files_list = files_list[
                     :int(np.floor(-1 * val_fraction * len(files_list)))
                     ]

    # Initialize slices generator
    pretrain_slice = prepare_pretrain_slice(
        files_list,
        params_path,
        longest_sequence,
        spk_max,
        spk_min,
        basename_len=basename_len,
        replicate=replicate
    )

    while True:
        # Initialize batch
        main_input = np.empty((batch_size, longest_sequence, 44))
        src_spk_in = np.empty((batch_size, longest_sequence))
        trg_spk_in = np.empty((batch_size, longest_sequence))
        feedback_in = np.empty((batch_size, longest_sequence, 44))
        params_output = np.empty((batch_size, longest_sequence, 42))
        flags_output = np.empty((batch_size, longest_sequence, 2))
        sample_weights = np.empty((batch_size, longest_sequence, 1))

        for i in range(batch_size):
            (
                main_input[i, :, :],
                src_spk_in[i, :],
                trg_spk_in[i, :],
                feedback_in[i, :, :],
                params_output[i, :, :],
                flags_output[i, :, :],
                sample_weights[i, :]
            ) = next(pretrain_slice)

        yield (
            {
                'main_input': main_input,
                'src_spk_in': src_spk_in,
                'trg_spk_in': trg_spk_in,
                'feedback_in': feedback_in
            },
            {
                'params_output': params_output,
                'flags_output': flags_output
            },
            {'sample_weights': sample_weights}
        )


def prepare_pretrain_slice(
        files_list,
        params_path,
        longest_sequence,
        spk_max,
        spk_min,
        speakers_file='speakers.list',
        dtw_prob_file='dtw_probabilities.h5',
        basename_len=11,
        shuffle_files=True,
        replicate=True
):
    speakers = open(os.path.join(params_path, speakers_file), 'r').readlines()
    # Strip '\n' characters
    speakers = [line.split('\n')[0] for line in speakers]

    with File(os.path.join(params_path, dtw_prob_file), 'r') as f:
        # Save numbers and probabilities
        values = f['values'][:]
        probabilities = f['probabilities'][:]

        f.close()

    # Read all files to have them loaded in memory
    mcp_params = []
    lf0_params = []
    mvf_params = []
    uv_flags = []

    for basename in files_list:
        mcp_params.append(parse_file(40, basename + '.cc'))
        lf0_params.append(parse_file(1,  basename + '.lf0_log'))
        mvf_params.append(parse_file(1,  basename + '.i.fv'))
        uv_flags.append(parse_file(1,    basename + '.lf0_log.uv_mask'))

    # Initialize file indexes
    indexes = np.arange(len(files_list))

    while True:
        if shuffle_files:
            # Shuffle file indexs before each epoch
            np.random.shuffle(indexes)

        # Iterate over shuffled files
        for file_index in indexes:
            # Compute speaker index
            basename = files_list[file_index]

            spk_index = speakers.index(
                str(basename[-1 * (basename_len + 3):-1 * (basename_len + 1)])
            )

            # # Read parameters
            # mcp_params = parse_file(40, basename + '.cc')
            # lf0_params = parse_file(1, basename + '.lf0_log')
            # mvf_params = parse_file(1, basename + '.i.fv')
            # uv_flags = parse_file(1, basename + '.lf0_log.uv_mask')

            # Get max and min values for each speaker
            src_spk_max = spk_max[spk_index, :]
            src_spk_min = spk_min[spk_index, :]

            # Maxmin normalize
            src_normalized = (np.concatenate((
                mcp_params[file_index],
                lf0_params[file_index],
                mvf_params[file_index]
            ), axis=1) - src_spk_min) / (src_spk_max - src_spk_min)

            # One-hot encode the speaker indexes
            spk_index_vector = np.repeat(
                spk_index,
                lf0_params[file_index].shape[0],
                axis=0
            )

            # Construct End-Of-Sequence flags
            eos_flags = np.zeros(lf0_params[file_index].shape[0])
            eos_flags[-1] = 1

            # Construct sequence "slice"
            seq_params = np.concatenate((
                src_normalized,
                uv_flags[file_index],
                np.reshape(eos_flags, (-1, 1)),
                np.reshape(spk_index_vector, (-1, 1)),
                np.reshape(spk_index_vector, (-1, 1)),
            ), axis=1)

            if replicate:
                # Replicate frames with dtw probabilities
                # TODO Change the function so it takes seq_params as separate args
                (src_res, trg_res, _, trg_mask) = replicate_frames(
                    seq_params,
                    longest_sequence,
                    values,
                    probabilities
                )
            else:
                src_res = np.concatenate((seq_params, np.zeros((
                    longest_sequence - seq_params.shape[0],
                    seq_params.shape[1]
                ))))
                trg_res = np.concatenate((seq_params[:, 0:44], np.zeros((
                    longest_sequence - seq_params.shape[0],
                    44
                ))))
                trg_mask = np.concatenate((
                    np.ones((seq_params.shape[0], 1)),
                    np.zeros((
                        longest_sequence - seq_params.shape[0],
                        1
                    ))
                ))

            # Prepare feedback data
            feedback_data = np.roll(trg_res, 1, axis=0)
            feedback_data[0, :] = 0

            # Return slice frames
            # print('Sliced ' + basename)
            yield (
                src_res[:, 0:44],
                src_res[:, 44:45].reshape((-1)),
                src_res[:, 45:46].reshape((-1)),
                feedback_data,
                trg_res[:, 0:42],
                trg_res[:, 42:44],
                trg_mask
            )
