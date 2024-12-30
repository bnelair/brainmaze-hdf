
import pandas as pd
import numpy as np

def get_data_segments(x):
    """
    Identifies contiguous NaN and non-NaN segments in an array.

    Args:
      x: The input array.

    Returns:
      A pandas DataFrame with columns 'start', 'end', and 'type',
      where 'type' indicates whether the segment is 'data' or 'nan'.
    """

    nans = np.isnan(x)
    segments = []

    if nans[0]:
        type_0 = 'nan'
    else:
        type_0 = 'data'

    segments.append({'start': 0, 'end': 0, 'type': type_0})

    for i in range(0, len(nans)-1):

        if nans[i] and not nans[i+1]: # beginning of data
            segments[-1]['end'] = i
            segments.append({'start': i+1, 'end': i+1, 'type': 'data'})

        elif not nans[i] and nans[i+1]: # beginning of nan
            segments[-1]['end'] = i
            segments.append({'start': i+1, 'end': i+1, 'type': 'nan'})

    segments[-1]['end'] = len(nans) - 1
    return pd.DataFrame(segments)


def create_block_indexes(data, block_size):
    """
    Generates block indexes for the given data, splitting blocks at NaN values.

    Args:
      data: The data array.
      block_size: The default block size.

    Returns:
      A list of dictionaries, where each dictionary represents a block with 'start_idx' and 'end_idx'.
    """

    df_segments = get_data_segments(data)

    # Filter only data segments
    df_data_segments = df_segments[df_segments['type'] == 'data']

    indexes = []
    for _, row in df_data_segments.iterrows():
        start = row['start']
        end = row['end'] + 1  # Include the last element of the segment
        for i in range(start, end, block_size):
            block_end = min(i + block_size, end)
            indexes.append({'start_idx': i, 'end_idx': block_end - 1})

    if len(indexes) == 0:
        return pd.DataFrame(columns=['start_idx', 'end_idx'])

    return pd.DataFrame(indexes)


def reinterpolate(x1, y1, x2):
    """
    Reinterpolates signal y1 with timestamps to y2 with samples x2.


    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """

    y2 = np.interp(
        x2,
        x1,
        y1,
        left=np.nan,
        right=np.nan
    )

    nans = np.isnan(y2)
    x2_c = x2[~nans]
    y2 = y2[~nans]

    return x2_c, y2



def get_involved_intervals(intervals: np.ndarray[int, int], start_uutc: float, end_uutc: float) -> pd.DataFrame:
    '''
    Returns indexes of the segments that are involved in the read interval.
    Input is a ndarray with [n_segments, (start, end)], returns array of indexes True or False

    This function returns a DataFrame with the
    :param seg_metadata:
    :param start_uutc:
    :param end_uutc:
    :return:
    '''

    # take only segments that
    # A) start is within the read interval
    # 0 0 0 0 0>1 1 1 1 1

    # B) end is within the read interval
    # 1 1 1 1 1<0 0 0 0 0

    # C) segment is completely within the read interval
    # 0 0 0>1 1 1 1<0 0 0

    # D) segment completely contains the read interval
    #>1 1 1 1 1 1 1 1 1 1<

    cond_a = (start_uutc >= intervals[:, 0]) & (start_uutc < intervals[:, 1])

    cond_b = (end_uutc >= intervals[:, 0]) & (end_uutc <= intervals[:, 1])

    cond_c = (start_uutc <= intervals[:, 0]) & (end_uutc >= intervals[:, 1])

    cond_d = (start_uutc >= intervals[:, 0]) & (end_uutc <= intervals[:, 1])

    cond = cond_a | cond_b | cond_c | cond_d

    return cond




