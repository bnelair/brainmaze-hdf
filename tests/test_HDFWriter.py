
import os
from turtledemo.penrose import start

import pytest
import h5py
import numpy as np
import scipy.signal as signal
import pandas as pd
from typing import List, Tuple

from datetime import datetime

from brainmaze_hdf.io import BrainmazeHDFWriter

from brainmaze_hdf.utils import get_data_segments,create_block_indexes

from conftest import temp_dir

def test_BrainMazeHDFWriter_create_channel(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')
    writer = BrainmazeHDFWriter(path)

    writer._open_write()

    chname = 'test_channel'

    writer._create_channel(chname)

    assert chname in writer.channels

    writer._close()

def test_BrainMazeHDFWriter_create_segment(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')
    writer = BrainmazeHDFWriter(path)

    chname = 'test_channel'

    writer._open_write()
    writer._create_channel(chname)

    start_uutc = int(datetime.now().timestamp() * 1e6)
    fsamp = 200

    seg = writer._create_segment(chname, start_uutc, fsamp)

    metadata = writer._get_channel_segment_metadata(chname)

    assert metadata.shape[0] == 1
    assert metadata.iloc[0]['start_uutc'] == start_uutc
    assert metadata.iloc[0]['end_uutc'] == start_uutc
    assert metadata.iloc[0]['fsamp'] == fsamp

    writer._close()

def test_BrainMazeHDFWriter_write_block(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')
    wrt = BrainmazeHDFWriter(path)

    chname = 'test_channel'

    wrt._open_write()
    wrt._create_channel(chname)

    start_uutc = int(datetime.now().timestamp() * 1e6)
    fsamp = 200
    x = np.random.randn(fsamp*60)
    end_uutc = int(start_uutc + (1e6 * x.shape[0] / fsamp))

    seg = wrt._create_segment(chname, start_uutc, fsamp)

    wrt._write_block(
        chname,
        seg,
        x,
        start_uutc,
        fsamp,
    )

    metadata = wrt._get_channel_segment_metadata(chname)

    assert metadata.iloc[0]['start_uutc'] == start_uutc
    assert metadata.iloc[0]['end_uutc'] == end_uutc
    assert metadata.iloc[0]['fsamp'] == fsamp

    indexes = np.array(wrt._session[chname][seg]['block_meta'])
    assert indexes.shape[0] == 1
    assert indexes[0][0] == start_uutc
    assert indexes[0][1] == end_uutc
    assert indexes[0][2] == x.shape[0]

    block = wrt._session[chname][seg].keys()
    block = [bn for bn in block if bn != 'block_meta'][0]

    x_read = wrt._session[chname][seg][block][()]
    assert np.all(x == x_read)

    wrt._close()

def test_BrainMazeHDFWriter_write(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')

    signal_len_s = 60
    fs = 200
    x = np.random.randn(signal_len_s * fs)
    start_uutc = int(datetime.now().timestamp() * 1e6)
    end_uutc = int(start_uutc + (1e6 * x.shape[0] / fs))
    chname = 'test_channel'
    start_read = start_uutc + (15*1e6)
    end_read = start_uutc + (45*1e6)

    wrt = BrainmazeHDFWriter(path)
    wrt.block_size = 5
    wrt._open_write()
    wrt.write_data(chname, x, fs, start_uutc)

    segment = wrt._get_channel_segment_metadata(chname)
    seg = segment.iloc[0]['segment_name']

    assert segment.iloc[0]['start_uutc'] == start_uutc
    assert segment.iloc[0]['end_uutc'] == end_uutc
    assert segment.iloc[0]['fsamp'] == fs

    indexes = np.array(wrt._session[chname][seg]['block_meta'])
    blocks = [fn for fn in wrt._session[chname][seg].keys() if fn != 'block_meta']

    involved_blocks_full = wrt._get_involved_blocks(chname, seg, start_uutc, end_uutc)
    involved_blocks_part = wrt._get_involved_blocks(chname, seg, start_read, end_read)

    assert np.all(np.array(blocks) == involved_blocks_full['block_name'].to_numpy())
    assert involved_blocks_full.__len__() == blocks.__len__()
    assert involved_blocks_part.__len__() < involved_blocks_full.__len__()

    x_read_full = wrt.read_data(chname, start_uutc, end_uutc)
    assert np.all(x_read_full == x)

    x_read_part = wrt.read_data(chname, start_read, end_read)
    assert np.all(x_read_part == x[int((start_read-start_uutc)/1e6*fs):int((end_read-start_uutc)/1e6*fs)])

    wrt._close()

def test_BrainMazeHDFWriter_write_nans_in_values(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')

    nan_segments_samples = [
        (0, 5),
        (10, 15),
        (20, 25),
        # (30, 31),
    ]

    signal_len_s = 60
    fs = 200

    x = np.random.randn(signal_len_s * fs)
    for start, end in nan_segments_samples:
        x[start:end+1] = np.nan

    # start_uutc = int(datetime.now().timestamp() * 1e6)
    start_uutc = 0

    end_uutc = int(start_uutc + (1e6 * x.shape[0] / fs))
    chname = 'test_channel'
    start_read = start_uutc + (15 * 1e6)
    end_read = start_uutc + (45 * 1e6)

    wrt = BrainmazeHDFWriter(path)
    wrt.block_size = fs
    wrt._open_write()
    wrt.write_data(chname, x, fs, start_uutc)


    x_read = wrt.read_data(chname, start_uutc, end_uutc)
    data_segments_read = get_data_segments(x_read)
    nan_segments_read_df = data_segments_read[data_segments_read['type'] == 'nan']
    nan_segments_read = [(int(row['start']), int(row['end'])) for _, row in nan_segments_read_df.iterrows()]

    assert nan_segments_samples == nan_segments_read

    wrt._close()

def test_BrainMazeHDFWriter_write_discontinuities(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')

    signal_len_s = 60
    fs = 200

    chname = 'test_channel'

    nans_beginning_s = 10
    nans_end_s = 10
    nans_middle_s = 10

    x1 = np.random.randn(signal_len_s * fs)
    x2 = np.random.randn(signal_len_s * fs)

    start_uutc_1 = int(datetime.now().timestamp() * 1e6)
    end_uutc_1 = int(start_uutc_1 + (1e6 * x1.shape[0] / fs))

    start_uutc_2 = end_uutc_1 + (1e6 * nans_middle_s)
    end_uutc_2 = int(start_uutc_2 + (1e6 * x2.shape[0] / fs))

    start_read = start_uutc_1 - (nans_beginning_s * 1e6)
    end_read = end_uutc_2 + (nans_end_s * 1e6)

    nan_segments_samples = [
        (0, nans_beginning_s * fs - 1),
        (int((end_uutc_1 - start_read) / 1e6 * fs), int((start_uutc_2 - start_read) / 1e6 * fs) - 1),
        (int((end_uutc_2 - start_read) / 1e6 * fs), int((end_read - start_read) / 1e6 * fs) - 1)
    ]

    wrt = BrainmazeHDFWriter(path)
    wrt.block_size = fs
    wrt._open_write()

    wrt.write_data(chname, x1, fs, start_uutc_1)
    wrt.write_data(chname, x2, fs, start_uutc_2)

    x_read = wrt.read_data(chname, start_read, end_read)
    nans = np.isnan(x_read)
    segments_read = get_data_segments(x_read)
    nan_segments_read = [(int(row['start']), int(row['end'])) for _, row in segments_read.iterrows() if row['type'] == 'nan']

    assert nan_segments_samples == nan_segments_read

    wrt._close()

def test_BrainMazeHDFWriter_read_interpolation(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')

    signal_len_s = 1
    fs = 20

    chname = 'test_channel'

    nans_beginning_s = 10
    nans_end_s = 10
    nans_middle_s = 10

    x1 = np.random.randn(signal_len_s * fs)
    x2 = np.random.randn(signal_len_s * fs)

    start_uutc_1 = int(datetime.now().timestamp() * 1e6) - 3600e6
    end_uutc_1 = int(start_uutc_1 + (1e6 * x1.shape[0] / fs))

    start_uutc_2 = end_uutc_1 + (1e6 * nans_middle_s)
    end_uutc_2 = int(start_uutc_2 + (1e6 * x2.shape[0] / fs))


    wrt = BrainmazeHDFWriter(path)
    wrt.block_size = 5
    wrt._open_write()

    wrt.write_data(chname, x1, fs, start_uutc_1)
    wrt.write_data(chname, x2, fs, start_uutc_2)

    start_read_interpolated_1 = start_uutc_1 - ((0.5/fs)*1e6)
    end_read_interpolated_1 = start_read_interpolated_1 + (end_uutc_2 - start_uutc_1)

    x_read_1 = wrt.read_data(chname, start_read_interpolated_1, end_read_interpolated_1)
    nans_1 = np.isnan(x_read_1)
    nan_segments_read_1 = get_data_segments(x_read_1)

    assert nans_1[0]
    assert not nans_1[1]
    assert not nans_1[-1]

    start_read_interpolated_2 = start_uutc_1 + ((0.5/fs)*1e6)
    end_read_interpolated_2 = start_read_interpolated_2 + (end_uutc_2 - start_uutc_1)

    x_read_2 = wrt.read_data(chname, start_read_interpolated_2, end_read_interpolated_2)
    nans_2 = np.isnan(x_read_2)
    nan_segments_read_2 = get_data_segments(x_read_2)

    assert not nans_2[0]
    assert nans_2[-1]
    assert not nans_2[-2]

    wrt._close()


if __name__ == '__main__':
    pytest.main([__file__])












