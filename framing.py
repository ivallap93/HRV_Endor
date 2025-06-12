import numpy as np

def frame_signal(time_array, signal_array, frame_length_sec, overlap_sec):
    """
    Frame a 1D signal based on time with overlapping windows.

    Parameters:
    - time_array: array of timestamps (in seconds)
    - signal_array: array of values (same length as time_array)
    - frame_length_sec: window length in seconds
    - overlap_sec: overlap between frames in seconds

    Returns:
    - List of framed signal segments (as arrays)
    """
    frames = []
    start = time_array[0]
    end = time_array[-1]

    i = start
    while i + frame_length_sec <= end:
        mask = (time_array >= i) & (time_array < i + frame_length_sec)
        segment = signal_array[mask]
        if len(segment) > 1:
            frames.append(segment)
        i += frame_length_sec - overlap_sec

    return frames