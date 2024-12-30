
import os
from turtledemo.penrose import start

import pytest
import h5py
import numpy as np
import scipy.signal as signal
import pandas as pd
from typing import List, Tuple

from datetime import datetime

from brainmaze_hdf.io import BrainmazeHDFWriter, BrainmazeHDFReader

from brainmaze_hdf.utils import get_data_segments,create_block_indexes

from conftest import temp_dir

from tqdm import tqdm


from mef_tools import MefReader, MefWriter



def test_benchmark_meftoolas(temp_dir: str):
    path = os.path.join(temp_dir, 'test.mefd')

    signal_len_s = 24*3600
    # signal_len_s = 10
    fs = 8000
    processing_block_size = 20*60
    n_channels = 1
    block_size = fs

    x = np.random.randn(signal_len_s * fs)

    start_uutc = int(datetime.now().timestamp() * 1e6)
    end_uutc = int(start_uutc + (1e6 * x.shape[0] / fs))

    wrt = MefWriter(path)
    wrt.block_size = block_size

    for ch in range(n_channels):
        chname = f'ch_{str(ch).zfill(3)}'

        for ts_segment in tqdm(np.arange(0, x.shape[0], processing_block_size * fs)):
            s_idx = ts_segment
            e_idx = int(ts_segment + (processing_block_size * fs))

            if e_idx > x.shape[0]:
                e_idx = x.shape[0]

            wrt.write_data(x[s_idx:e_idx], channel=chname, start_uutc=start_uutc + (s_idx / fs * 1e6), sampling_freq=fs, precision=3, )


    # rdr = BrainmazeHDFReader(path)
    rdr = MefReader(path)

    for ch in rdr.channels:
        for s_ in tqdm(np.arange(start_uutc, end_uutc, processing_block_size*1e6)):
            e_ = s_ + (processing_block_size*1e6)

            idx_s = int((s_ - start_uutc) / 1e6 * fs)
            idx_e = int((e_ - start_uutc) / 1e6 * fs)

            if e_ > end_uutc:
                e_ = end_uutc
                idx_e = int((e_ - start_uutc) / 1e6 * fs)

            # x_ = rdr.read_data(ch, s_, e_)
            x_ = rdr.get_data(ch, s_, e_)

            assert np.allclose(x_, x[idx_s : idx_e], 3)


def test_benchmark_BrainMazeHDFWriter(temp_dir: str):
    path = os.path.join(temp_dir, 'test.h5')

    signal_len_s = 24*3600
    # signal_len_s = 10
    fs = 8000
    processing_block_size = 20*60
    n_channels = 1
    block_size = fs

    x = np.random.randn(signal_len_s * fs)

    start_uutc = int(datetime.now().timestamp() * 1e6)
    end_uutc = int(start_uutc + (1e6 * x.shape[0] / fs))

    wrt = BrainmazeHDFWriter(path)
    wrt.block_size = block_size
    wrt.compression_format = None

    print(wrt.compression_format)

    for ch in range(n_channels):
        chname = f'ch_{str(ch).zfill(3)}'

        for ts_segment in tqdm(np.arange(0, x.shape[0], processing_block_size * fs)):
            s_idx = ts_segment
            e_idx = int(ts_segment + (processing_block_size * fs))

            if e_idx > x.shape[0]:
                e_idx = x.shape[0]

            wrt.write_data(chname, x[s_idx:e_idx], fs, start_uutc + (s_idx / fs * 1e6), )

    wrt._close()


    rdr = BrainmazeHDFReader(path)

    for ch in rdr.channels:
        for s_ in tqdm(np.arange(start_uutc, end_uutc, processing_block_size*1e6)):
            e_ = s_ + (processing_block_size*1e6)

            idx_s = int((s_ - start_uutc) / 1e6 * fs)
            idx_e = int((e_ - start_uutc) / 1e6 * fs)

            if e_ > end_uutc:
                e_ = end_uutc
                idx_e = int((e_ - start_uutc) / 1e6 * fs)

            x_ = rdr.read_data(ch, s_, e_)

            assert np.all(x_ == x[idx_s : idx_e])





if __name__ == '__main__':
    pytest.main([__file__])












