import sys

import scipy
import wfdb
import numpy as np
import os
from src.ctg_utils import get_all_recno, parse_meta_comments
from src.basic_denoise import calculate_valid_segments, get_valid_segments
import scipy.io as sio
"""
# source code from:  https://github.com/williamsdoug/CTG_RP/tree/master
# source data from: https://www.physionet.org/content/ctu-uhb-ctgdb/1.0.0/
"""

mat_contents = sio.loadmat(('D:\\workspace\\ECG\\CTG\\readdata\\'+'indexCTUdata.mat'))
hypoxia_id = mat_contents['hypoxia_id'] - 1
hypoxiaAsuspi_id = mat_contents['hypoxiaAsuspi_id'] - 1
norm_id =  mat_contents['norm_id'] - 1 # matlab index to python index
normNsuspicious_id =  mat_contents['normNsuspicious_id'] - 1 # matlab index to python index
norm_id=norm_id.flatten()
normNsuspicious_id = normNsuspicious_id.flatten()
hypoxia_id = hypoxia_id.flatten()
hypoxiaAsuspi_id = hypoxiaAsuspi_id.flatten()

RECORDINGS_DIR = 'ctuchbdataset\\'
inx = sorted(get_all_recno(RECORDINGS_DIR))

# get the list of signal id, that it has some good quality
def getvalidinx(idhere):
    # idhere-> list of candidates, where it classified to some category, but its quality might not be good.
    validlist = []
    for  pid in idhere:
        recno = inx[pid]
        recno_full = os.path.join(RECORDINGS_DIR, recno)
        # print('\nRecord: {}'.format(recno))
        all_sig, meta = wfdb.io.rdsamp(recno_full)

        orig_hr = all_sig[7200:, 0]  # we begin with the second 30 mins. that is 4*60*30 =  7200
        sig_hr = np.copy(orig_hr)
        ts = np.arange(len(sig_hr)) / 4.0

        # this is to get the valid segment-after filling missing values and other treat
        # selected_segments = get_valid_segments(orig_hr, ts, recno, verbose=True,
        #                                        # max_change=15, verbose_details=True
        #                                        )
        # this is only check the valid segment-without repairing
        dict_valid = calculate_valid_segments(orig_hr, ts, recno, verbose=False,
                                               # max_change=15, verbose_details=True
                                               )

        # print([values for values in dict_valid.values()]) # id, total_valid_per, seg0_per, seg1_per_valid....
        vals = [values for values in dict_valid.values()]
        # vals[0]->id, vals[1]->total_valid_per, vals[2]-> seg0_valid_percet....,

        idsave = vals[0]
        vals.remove(vals[0])
        if isinstance(vals, list):
            if max(vals) > 80:
                validlist.append(int(idsave))
        # if len(vals) < 5:
        #     if vals[1]> 65.0: #75.0: # valid percetnt in total,
        #         if vals[2] > 75.0: #85.0:
        #             # print([values for values in dict_valid.values()])
        #             if len(vals) == 3:
        #                 print(int(vals[0]))
        #                 validlist.append(int(vals[0]))
        #             elif vals[3] > 75.0: #85.0:
        #                 print(int(vals[0]))
        #                 validlist.append(int(vals[0]))
        # continue

    return validlist

# generate signals segment into data, 'limit'-> how many data want to get
def generate_numpy_signals(recordings_dir, n_dec=4, clip_stage_II=True,
                       max_seg_min=10, policy='late_valid',
                       results = None,
                       signal_dir='',
                       y_truth=None, my_y=None, verbose=False, my_x=None,
                       limit=-1, idlist_me=None):
    assert policy in ['best_quality', 'early_valid', 'late_valid']

    if signal_dir and not os.path.exists(signal_dir):
        os.mkdir(signal_dir)

    max_seg = int(max_seg_min * 60 * 4)  # convert to samples


    # inx = sorted(get_all_recno(recordings_dir))
    for recno in idlist_me: # idlist_me-> the signal id that satisfied the quality test
        limit -= 1
        if limit == 0:
            break

        recno_full = os.path.join(recordings_dir, str(recno))
        all_sig, meta = wfdb.io.rdsamp(recno_full)
        meta = parse_meta_comments(meta['comments'])
        if verbose:
            print('\nRecord: {}  Samples: {}   Duration: {:0.1f} min   Stage.II: {} min'.format(
                recno, all_sig.shape[0], all_sig.shape[0] / 4 / 60, meta['Delivery']['II.stage']))

        sig_hr = all_sig[:, 0]
        if clip_stage_II and meta['Delivery']['II.stage'] != -1:
            idx = int(meta['Delivery']['II.stage'] * 60 * 4)
            sig_hr = sig_hr[:-idx]
        ts = np.arange(len(sig_hr)) / 4.0  # useful when showing the image

        # select segment with lowest error rate
        selected_segments = get_valid_segments(sig_hr, ts, recno, verbose=False)

        if len(selected_segments) == 0:
            continue

        if policy == 'best_quality':
            selected_segments = sorted(selected_segments, key=lambda x: -x['pct_valid'])
        elif policy == 'early_valid':
            selected_segments = sorted(selected_segments, key=lambda x: x['seg_start'])
        elif policy == 'late_valid':
            selected_segments = sorted(selected_segments, key=lambda x: -x['seg_end'])

        if y_truth == 0 or y_truth == 2:
            seg = selected_segments[0]
            seg_hr = seg['seg_hr']

            if policy == 'late_valid':
                selected_hr = seg_hr[-max_seg:]
            else:
                selected_hr = seg_hr[:max_seg]# this is what you want to find clip-the-data

            if n_dec > 1: #downsample
                selected_hr = scipy.signal.decimate(selected_hr, n_dec)
            # Downsample the signal after applying an anti-aliasing filter.
            my_x.append(selected_hr)
            my_y.append(np.array([y_truth]))
            results[recno] = {'names': recno, 'outcome': meta['Outcome']}
        else:
            for sg in range(len(selected_segments)):  # this is for hypoxia class or hypoxia suspisious, they have less samples
                if selected_segments[sg]['pct_valid'] > 0.97:
                    seg = selected_segments[sg]
                    seg_hr = seg['seg_hr']
                    selected_hr = seg_hr[-max_seg:]
                    selected_hr = scipy.signal.decimate(selected_hr, n_dec)
                    my_x.append(selected_hr)
                    my_y.append(np.array([y_truth]))
                    results[recno] = {'names': recno, 'outcome': meta['Outcome']}
    # print(my_y)
    return my_x, my_y, results


if __name__ == "__main__":
    my_x = []
    my_y = []
    result ={}
    # validlist = getvalidinx(hypoxia_id)
    names = ['norm_id.npy', 'hypoxia_id.npy', 'normNsuspicious_id.npy', 'hypoxiaAsuspi_id.npy']

    with open(('files/'+names[0]), 'rb') as f:
        norm_id_left = np.load(f)
    with open(('files/'+names[1]), 'rb') as f:
        hypoxia_id_left = np.load(f)
    with open(('files/'+names[2]), 'rb') as f:
        normsus_id_left = np.load(f)
    with open(('files/'+names[3]), 'rb') as f:
        hypoxiasus_id_left = np.load(f)

    # set(normsus_id_left) - set(norm_id_left) # do not use set method to remove-exclusive elements
    normsusp_id = np.setxor1d(normsus_id_left, norm_id_left)
    hypoxiasusp_id =  np.setxor1d(hypoxiasus_id_left, hypoxia_id_left)

    keys = ['norm_id', 'hypoxia_id', 'normNsuspicious_id', 'hypoxiaAsuspi_id']
    parmas = {'norm_id': norm_id_left,
              'hypoxia_id': hypoxia_id_left,
              'normNsuspicious_id': normsusp_id,
              'hypoxiaAsuspi_id': hypoxiasusp_id
              }
    for id, key in enumerate(keys):
        if id == 0 or id == 1:
            truthy = 0
        else:
            truthy = 1
        my_x, my_y, result = generate_numpy_signals(RECORDINGS_DIR, n_dec=4, clip_stage_II=True,
                               max_seg_min=10, policy='best_quality',
                               results= result,
                               signal_dir='data/',
                               y_truth=truthy, my_y=my_y, verbose=False, my_x=my_x,
                               limit=-1, idlist_me=parmas[key])


# https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
    print(my_y)
    print(len(my_y))
    print(np.unique(my_y, return_counts=True))

    # (array([0, 1, 2, 3]), array([39, 17, 45, 21], dtype=int64)) summary
    with open('data_x.npy', 'wb') as f:
        np.save(f, my_x)
    with open('data_y.npy', 'wb') as f:
        np.save(f, my_y)
    with open('details_xy.npy', 'wb') as f:
        np.save(f, result)