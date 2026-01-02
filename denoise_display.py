import sys

import wfdb
import numpy as np
import os
from src.ctg_utils import get_all_recno, parse_meta_comments
from src.basic_denoise import calculate_valid_segments, get_valid_segments, draw_valid_segments
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

RECORDINGS_DIR = 'C:\\Users\\16499\\Downloads\\ctuchbdataset\\'
inx = sorted(get_all_recno(RECORDINGS_DIR))

validlist = []
def getinx(idhere):
    for sid, pid in enumerate(idhere):
        if sid < 8:
            continue
        recno = inx[pid]
        recno_full = os.path.join(RECORDINGS_DIR, recno)
        # print('\nRecord: {}'.format(recno))
        all_sig, meta = wfdb.io.rdsamp(recno_full)
        # print('nSamples: {}'.format(all_sig.shape[0]))
        # pprint(meta['comments'])

        orig_hr = all_sig[7200:, 0]  # we begin with the second 30 mins. that is 4*60*30 =  7200
        sig_hr = np.copy(orig_hr)
        sig_uc = all_sig[7200:, 1]
        ts = np.arange(len(sig_hr)) / 4.0

        selected_segments = draw_valid_segments(orig_hr, ts, recno, verbose=True,
                                               max_change=15, verbose_details=True
                                               )
        # this is to get the valid segment-after filling missing values and other treat
        selected_segments = get_valid_segments(orig_hr, ts, recno, verbose=True,
                                               max_change=15, verbose_details=True
                                               )
        # this is only check the valid segment-without repairing
        dict_valid = calculate_valid_segments(orig_hr, ts, recno, verbose=False,
                                               max_change=15, verbose_details=True
                                               )

        # print([values for values in dict_valid.values()]) # id, total_valid_per, seg0_per, seg1_per_valid....
        vals = [values for values in dict_valid.values()]
        # vals[0]->id, vals[1]->total_valid_per, vals[2]-> seg0_valid_percet....,
        idsave = vals[0]
        vals.remove(vals[0])
        if isinstance(vals, list):
            if max(vals) > 90:  # saved instances
                validlist.append(int(idsave))

        # continue
        break;


names = ['norm_id.npy', 'hypoxia_id.npy', 'normNsuspicious_id.npy', 'hypoxiaAsuspi_id.npy']
key = [  'hypoxia_id', 'normNsuspicious_id', 'norm_id','hypoxiaAsuspi_id']
parmas = {'norm_id': norm_id,
'hypoxia_id': hypoxia_id,
 'normNsuspicious_id': normNsuspicious_id,
 'hypoxiaAsuspi_id': hypoxiaAsuspi_id
}

if __name__ == "__main__":
    for use in range(4):
        nameid = use
        # this is probabaly the final figure draw.
        getinx(parmas[key[nameid]])  # hypoxia_id

        with open(names[nameid], 'wb') as f:
            np.save(f, validlist)

        with open(names[nameid], 'rb') as f:
            a = np.load(f)

        for  pid in a:
            print(pid)
        print('lenth  ', len(a))