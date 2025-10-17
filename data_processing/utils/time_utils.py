from datetime import datetime
import numpy as np

def str2timestamp(str_timestamp):
    if isinstance(str_timestamp, bytes):
        str_timestamp = str_timestamp.decode()
    timestamp = datetime.strptime(str_timestamp, "%Y-%m-%d %H:%M:%S.%f")
    return timestamp

def select_period(src_timestamp: np.ndarray, start: datetime, stop: datetime):
    """ Select the period when happend in the [start, end].
    """
    return np.where((src_timestamp >= start) & (src_timestamp <= stop))[0]


extract_sortkey = lambda path : datetime.strptime(path.split("/")[-1].split("_streamLog")[0], "%Y-%m-%d_%H-%M-%S")