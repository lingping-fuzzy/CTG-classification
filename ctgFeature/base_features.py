import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, argrelextrema
# equations: https://matteogambera.medium.com/how-to-extract-features-from-signals-15e7db225c15

def log_features(X, logger, feature_func):
    feats = X.columns
    logger.info("****** Feature Creation **********")
    logger.info("Created features with {}".format(feature_func))
    logger.info("Features include {}".format(feats))
    logger.info("Feature describe")
    logger.info("\n{}".format(X.describe()))


def make_y_df(n_size, s_size):
    y_array = np.zeros(n_size + s_size, dtype=int)
    y_array[-s_size:] = 1
    return y_array


def base_feat(X_in, logger, dsetsize=None):
    from scipy import stats
    X = pd.DataFrame()
    X['mean'] = X_in.mean()
    X['std'] = X_in.std()#-> importance
    X['mad'] = (X_in - X_in.median()).abs().median()#-> importance
    # X['mad'] = stats.median_abs_deviation(X_in)
    X['diff'] = X_in.diff().sum()#-> importance
    X['skew'] = X_in.skew()
    X['pulseindicator'] = X_in.abs().max() / X_in.mean()#-> importance
    X['kurtosis'] = X_in.kurtosis() #-> importance
    X['cumsum'] = X_in.cumsum(axis=0).iloc[-1].values

    log_features(X, logger, "base_feat")

    if dsetsize is not None:
        X = X.sample(dsetsize)
    return X


def smart_scale(my_vector):
    my_vector = my_vector - np.median(my_vector) # center

    if my_vector.max() < 1.0:
        return my_vector
    else:
        return my_vector / np.max(my_vector) # scale



def gen_spect(sample):
    sample = sample-np.median(sample)
    ps = np.abs(np.fft.fft(sample)) ** 2
    time_step = 0.25
    zoom = 2
    freqs = np.fft.fftfreq(sample.size, time_step)
    idx = np.argsort(freqs)
    half_way = len(freqs) // 2
    ps2 = 2 * ps[:half_way]

    ps2 = smart_scale(ps2)

    bin = []
    bin.append(np.sum(ps2[2:5]))   # lowest frequency bin excluding near DC
    bin.append(np.sum(ps2[5:12]))  # second lowest bin of frequencies
    bin.append(np.sum(ps2[12:30])) # mid segment of the spectral energy
    bin.append(np.sum(ps2[30:])) # high segment of the spectral energy
    ''''
    bin.append(np.sum(ps2[90:120]))
    bin.append(np.sum(ps2[120:150]))
    bin.append(np.sum(ps2[150:180]))
    bin.append(np.sum(ps2[180:210]))
    bin.append(np.sum(ps2[210:]))
    '''
    return bin


def spectrum_feat(X_in, logger, dsetsize=None):
    #dc, low, mid, high, uh, A, B, C, D = [], [], [], [], [], [], [], [], []
    dc, low, mid, high = [], [], [], []
    for column in X_in.columns:
        bins = gen_spect(X_in[column].values)
        dc.append(bins[0])
        low.append(bins[1])
        mid.append(bins[2])
        high.append(bins[3])

    X = pd.DataFrame(np.array([dc, low, mid, high]).T, columns=['SP_ULF','SP_VLF', 'SP_LF','SP_HF'])

    #X = pd.DataFrame(np.array([dc, low, mid, high, uh, A, B, C, D]).T,
    #                 columns=['SP_ULF', 'SP_VLF', 'SP_LF', 'SP_A', 'SP_B', 'SP_C', 'SP_D', 'SP_E', 'SP_F'])

    if dsetsize is not None:
        X = X.sample(dsetsize)

    log_features(X, logger, "spectrum_feat")
    return X


def autocorr_feat(X_in, logger, dsetsize=None):

    b, a = butter(3, 0.1, btype='low', analog=False) # 3-tap, 0.1 normalized low pass filter

    max_mins, min_sum, max_sum, min_median, max_median, min_std, max_std = [], [], [], [], [], [], []
    for i, column in enumerate(X_in.columns):
        if i>100:
            break
        sig = smart_scale(X_in[column].values) # zero center normalized or below 1.0 max
        lags, c, _, _ = plt.acorr(sig, maxlags=240) # autocorrelation in c with 240 lags (1mins, 60*4hz)
        plt.clf()
        c_filt = lfilter(b, a, c) # smooth the signal
        max_indx_list = argrelextrema(c_filt, np.greater)[0]
        min_indx_list = argrelextrema(c_filt, np.less)[0]
        maximums = c_filt[max_indx_list]
        minimums = c_filt[min_indx_list]
        max_mins.append(len(max_indx_list)+len(min_indx_list))
        min_sum.append(np.sum(minimums))
        max_sum.append(np.sum(maximums))
        if not minimums.any():
            min_median.append(0)
            min_std.append(0)
        else:
            min_median.append(np.median(minimums))
            min_std.append(np.std(minimums))
        max_median.append(np.median(maximums))
        max_std.append(np.std(maximums))

    X = pd.DataFrame(np.array([max_mins, min_sum, max_sum, min_median, max_median, min_std, max_std]).T,
                     columns=['MAX_MINS','MIN_SUM','MAX_SUM','MIN_MEDIAN', 'MAX_MEDIAN', 'MIN_STD','MAX_STD'])

    '''
    if dsetsize is not None:
        X = X.sample(dsetsize)
        y = y[X.index.values]
    '''
    log_features(X, logger, "autocorr_feat")
    return X