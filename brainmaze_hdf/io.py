
import os
# from turtledemo.penrose import start

from abc import ABC, abstractmethod

import h5py
import numpy as np
import scipy.signal as signal
import pandas as pd
from typing import List, Tuple

from brainmaze_hdf.utils import get_data_segments, create_block_indexes, reinterpolate, get_involved_intervals

class BrainmazeHDFSession(ABC):
    def __init__(self, path: str, block_size=1000, compression_format='gzip'):
        self.path = path

        self._session = None
        self._mode = None

        self.compression_format = compression_format
        self.block_size = block_size

    def _open_read(self):
        self._mode = 'r'
        self._session = h5py.File(self.path, 'r')

    def _open_write(self):
        if os.path.isfile(self.path):
            self._mode = 'a'
        else:
            self._mode = 'w'

        self._session = h5py.File(self.path, self._mode)

    def _close(self):
        self._session.close()

        self._mode = ''
        self._session = None

    def _create_channel(self, channel):
        if channel not in self._session:
            self._session.create_group(channel)

    def _get_channel_segment_metadata(self, channel: str) -> pd.DataFrame:
        # check if a segment exists
        segments = []
        for seg in self._session[channel]:
            segments += [{
                'segment_name': seg,
                'start_uutc': self._session[channel][seg].attrs['start_uutc'],
                'end_uutc': self._session[channel][seg].attrs['end_uutc'],
                'duration': self._session[channel][seg].attrs['end_uutc'] - self._session[channel][seg].attrs['start_uutc'],
                'fsamp': self._session[channel][seg].attrs['fsamp'],
                'ufact': self._session[channel][seg].attrs['ufact'],
            }]

        if len(segments):
            return pd.DataFrame(segments)

        return pd.DataFrame(columns=['segment_name', 'start_uutc', 'end_uutc', 'duration', 'fsamp', 'ufact'])

    def _create_segment(self, channel, start_uutc, fsamp: float, ufact: float = 1.0):
        segments_metadata = self._get_channel_segment_metadata(channel)
        # new segment must start later than the last segment

        seg_idx = len(segments_metadata)
        seg_name = f'seg_{str(seg_idx).zfill(4)}'
        start_uutc = int(start_uutc)

        if len(segments_metadata) > 0:
            last_end = segments_metadata['end_uutc'].max()
            if start_uutc <= last_end:
                raise ValueError(f"New segment must start after the last segment: {start_uutc} < {last_end}")

        self._session[channel].create_group(seg_name)
        self._session[channel][seg_name].attrs['start_uutc'] = start_uutc
        self._session[channel][seg_name].attrs['end_uutc'] = start_uutc
        self._session[channel][seg_name].attrs['fsamp'] = fsamp
        self._session[channel][seg_name].attrs['ufact'] = ufact

        self._session[channel][seg_name].create_dataset(
            'block_meta',
            data=np.array([]).reshape((0, 3)).astype(np.int64),
            maxshape=(None, 3),
            compression=self.compression_format
        )

        return seg_name

    def _write_block(self, channel: str, segment: str, data: np.ndarray, start_uutc: float, fsamp: float):
        # check if channel exists
        if channel not in self._session:
            self._create_channel(channel)

        start_uutc = int(np.round(start_uutc))

        # check if data is consistent with segment
        # print(start_uutc, self.session[channel][segment].attrs['end_uutc'])
        if start_uutc < self._session[channel][segment].attrs['end_uutc']:
            raise ValueError(f"Data must start after the last data block: {start_uutc} < {self._session[channel][segment].attrs['end_uutc']}")

        # write data
        self._session[channel][segment].create_dataset(
            str(start_uutc),
            data=data,
            compression=self.compression_format
        )

        end_uutc = int(start_uutc + (1e6 * len(data) / self._session[channel][segment].attrs['fsamp']))

        self._session[channel][segment].attrs['end_uutc'] = end_uutc # updating a segment end based on the last data block.

        idx = np.array([
            start_uutc,
            end_uutc,
            len(data)
        ], dtype=np.int64).reshape(1, 3)

        n_blocks = self._session[channel][segment]['block_meta'].shape[0]
        self._session[channel][segment]['block_meta'].resize((n_blocks + 1, 3), )
        self._session[channel][segment]['block_meta'][-1] = idx


    def _get_involved_segments(self, channel: str, start_uutc: float, end_uutc: float) -> pd.DataFrame:
        '''
        Get the segments that are involved in the read operation. This function returns a DataFrame with the
        :param seg_metadata:
        :param start_uutc:
        :param end_uutc:
        :return:
        '''

        seg_metadata = self._get_channel_segment_metadata(channel)
        intervals = np.array([seg_metadata['start_uutc'], seg_metadata['end_uutc']]).T
        idxes = get_involved_intervals(intervals, start_uutc, end_uutc)
        return seg_metadata.loc[idxes]

    def _get_involved_blocks(self, channel: str, segment: str, start_uutc: float, end_uutc: float) -> pd.DataFrame:
        '''
        Get the blocks that are involved in the read operation. This function returns a DataFrame with the
        :param seg_metadata:
        :param start_uutc:
        :param end_uutc:
        :return:
        '''

        segment_block_metadata = self._session[channel][segment]['block_meta'][()]
        segment_block_names = [str(fn) for fn in segment_block_metadata[:, 0]]

        intervals = segment_block_metadata[:, :2]
        cond = get_involved_intervals(intervals, start_uutc, end_uutc)

        involved_block_names = []
        involved_block_indexes = []
        for i, blck in enumerate(segment_block_names):
            if cond[i]:
                involved_block_names.append(blck)
                involved_block_indexes.append(i)

        involved_segments = []
        for blck_idx, blck_name in zip(involved_block_indexes, involved_block_names):
            start_block_uutc = segment_block_metadata[blck_idx, 0]
            end_block_uutc = segment_block_metadata[blck_idx, 1]

            involved_segments += [{
                'block_name': blck_name,
                'start_block_uutc': start_block_uutc,
                'end_block_uutc': end_block_uutc,
                'duration': end_block_uutc - start_block_uutc,
                'n_samples': segment_block_metadata[blck_idx, 2]
            }]

        if len(involved_segments):
            involved_segments = pd.DataFrame(involved_segments)
        else:
            involved_segments = pd.DataFrame(columns=['segment_name', 'start_block_uutc', 'end_block_uutc', 'duration', 'n_samples'])

        return involved_segments

    def _get_block_data(self, channel: str, segment: str, block, start_uutc: float, end_uutc: float) -> Tuple[np.ndarray, float]:
        # check if channel exists
        if channel not in self._session:
            raise ValueError(f"Channel {channel} does not exist")

        # check if segment exists
        if segment not in self._session[channel]:
            raise ValueError(f"Segment {segment} does not exist")

        # check if block exists
        if block not in self._session[channel][segment]:
            raise ValueError(f"Block {block} does not exist")

        # segment_block_names = [fn for fn in self.session[channel][segment] if fn != 'block_meta']
        # segment_block_metadata = self.session[channel][segment]['block_meta'][()]

        data = self._session[channel][segment][block][()]
        return data

    @property
    def channels(self) -> List[str]:
        return list(self._session.keys())

    def n_segments(self, channel: str) -> int:
        return len(self._session[channel])

    def write_data(self, channel: str, data: np.ndarray, fsamp: float, start_uutc: float, new_segment: bool = False):

        if not channel in self.channels:
            self._create_channel(channel)

        if self.n_segments(channel) == 0 or new_segment:
            segment = self._create_segment(channel, start_uutc, fsamp)
        else:
            segment = list(self._session[channel].keys())[-1]

        block_indexes = create_block_indexes(data, self.block_size)

        for i, row in block_indexes.iterrows():
            start_block_idx = row['start_idx']
            end_block_idx = row['end_idx']

            start_block_uutc = start_uutc + start_block_idx / fsamp * 1e6

            self._write_block(channel, segment, data[start_block_idx : end_block_idx+1], start_block_uutc, fsamp)

    def read_data(self, channel: str, start_uutc: float, end_uutc: float) -> Tuple[np.ndarray, float]:
        '''
        Read data from a channel between two timestamps. The data is returned as a numpy array and the
        sampling frequency is returned as a float.

        The data is read between segments, if overlapping and between blocks if the sampling rate is same.
        Otherwise an exception is raised. If there are parts of the data that are missing, the missing parts
        are filled with nans.

        :param channel:
        :param start_uutc:
        :param end_uutc:
        :return:
        '''

        # refactored writing to split blocks for nans.. TODO this func to redo the interpolation
        if not channel in self.channels:
            raise ValueError(f"Channel {channel} does not exist")

        seg_involved = self._get_involved_segments(channel, start_uutc, end_uutc)

        # all blocks must have same sampling rate
        if len(seg_involved['fsamp'].unique()) > 1:
            raise ValueError(
                f"Segments in the selected interval have different sampling rates: {seg_involved['fsamp'].unique()}"
            )

        fsamp = seg_involved.iloc[0]['fsamp']
        fsamp = float(fsamp)

        xt_outp = np.arange(start_uutc, end_uutc, int(1/fsamp*1e6))
        xd_outp = np.zeros(int((end_uutc - start_uutc)/1e6*fsamp)) + np.nan

        all_timestamps = []
        all_data = []

        for i, seg in seg_involved.iterrows():
            blocks_involved = self._get_involved_blocks(channel, seg['segment_name'], start_uutc, end_uutc)

            for j, block in blocks_involved.iterrows():
                block_start = block['start_block_uutc']
                block_end = block['end_block_uutc']
                block_name = block['block_name']

                # print(block_start, block_end)

                # Process the current block
                xd_block = self._get_block_data(channel, seg['segment_name'], block_name, block_start, block_end)
                xt_block = np.arange(block_start, block_end, int(1 / fsamp * 1e6))

                if len(all_timestamps):
                    if xt_block[0] == all_timestamps[-1][-1] + (1/fsamp*1e6):
                        all_timestamps[-1] = np.concatenate([all_timestamps[-1], xt_block])
                        all_data[-1] = np.concatenate([all_data[-1], xd_block])
                    else:
                        all_timestamps += [xt_block]
                        all_data += [xd_block]
                else:
                    all_timestamps += [xt_block]
                    all_data += [xd_block]

        # Interpolate all the data to the same timestamps
        for i, (xt, xd) in enumerate(zip(all_timestamps, all_data)):
            print(xt[0])
            xt_, xd_ = reinterpolate(xt, xd, xt_outp)
            idx1 = int((xt_[0] - start_uutc) / 1e6 * fsamp)
            idx2 = int((xt_[-1] - start_uutc) / 1e6 * fsamp)
            xd_outp[idx1:idx2+1] = xd_

        return xd_outp

class BrainmazeHDFWriter(BrainmazeHDFSession):
    def __init__(self, path: str, block_size=1000, compression_format='gzip', overwrite=False):
        super().__init__(path, block_size, compression_format)
        self._open_write()

class BrainmazeHDFReader(BrainmazeHDFSession):
    def __init__(self, path: str):
        super().__init__(path)
        self._open_read()







