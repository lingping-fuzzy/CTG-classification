import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Create np array with each row presenting a windowing vector length v_len
def make_windows(v_len):
    win_name = ['blackman', 'hamming','hann','flattop', 'blackmanharris', 'tukey']
    windows = []

    for window in win_name[:-1]:
        windows.append(signal.get_window(window, v_len))

    windows.append(signal.windows.tukey(v_len, alpha=0.1))

    return np.array(windows), win_name

