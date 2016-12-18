# Utils module
# coding: utf-8

# Copyright © 2016 Albert Aparicio
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import division

import numpy as np


def kronecker_delta(x):
    if x == 0:
        return 1
    else:
        return 0


def apply_context(input_matrix, context_size):
    """Currently this function only works for:
        - Input matrix of Nx1 elements"""
    assert input_matrix.shape[0] == input_matrix.size

    # Reshape into (N,1) array, avoiding shapes like (N,)
    input_matrix = input_matrix.reshape((-1, 1))

    # Replicate matrix 'context_size' times
    replicated = input_matrix.repeat(2 * context_size + 1, axis=1)

    # Roll context elements
    for i in np.arange(0, 2 * context_size + 1):
        replicated[:, i] = np.roll(replicated[:, i], context_size - i)

        # TODO Zero out the out-of-context samples
        if (context_size - i) > 0:
            replicated[0:context_size - i, i] = 0

        elif (context_size - i) < 0:
            # replicated[0:context_size - i, i] = 0
            replicated[
            replicated.shape[0] + (context_size - i):replicated.shape[0], i] = 0

    return replicated


def reshape_lstm(a, tsteps, data_dim):
    """Zero-pad and reshape the input matrix 'a' so it can be fed
    into a stateful LSTM-based RNN"""
    # Compute the amount of zero-vectors that need to be added to the matrix
    zpad_size = int((np.ceil(a.shape[0] / tsteps) * tsteps) - a.shape[0])

    # Initialize padding matrix
    zeros_matrix = np.zeros((zpad_size, a.shape[1]))

    # Concatenate 'a' and zero-padding matrix
    a = np.concatenate((a, zeros_matrix))

    # Reshape training and test data
    return a.reshape((int(a.shape[0] / tsteps), tsteps, data_dim))


def display_time(seconds, granularity=2):
    """ This function converts an amount of seconds into minutes, hours, etc.

    This code has been extracted from StackOverflow, posted by Ralph Bolton:
    http://stackoverflow.com/a/38222840

    As it has been posted after March 1, 2016, it is covered by the MIT License:
    http://meta.stackexchange.com/q/272956

    :param seconds: amount of seconds to be converted
    :param granularity: precision of the date display (see the original post)
    :return: converted time in weeks, days, hours, minutes and seconds,
             depending on the granularity
    """

    intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )

    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
        else:
            # Add a blank if we're in the middle of other values
            if len(result) > 0:
                result.append(None)
    return ', '.join([x for x in result[:granularity] if x is not None])
