import numpy as np
from base_features import base_feat, spectrum_feat, autocorr_feat
import logging
import pandas as pd
from scipy.io import loadmat
if __name__ == '__main__':

    with open('..//data_x.npy', 'rb') as f:
        my_x = np.load(f)
    with open('..//data_y.npy', 'rb') as f:
        my_y = np.load(f)

    logging.basicConfig(filename='log.text',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    feat_data = pd.DataFrame()
    for id in range(len(my_y)):
        print('this is ', id, '-------------------------------------')
        sig = my_x[id, :]
        X1 = base_feat(pd.DataFrame(sig), logger=logging) #8 feature
        X2 = spectrum_feat(pd.DataFrame(sig), logger=logging)#4 feature
        X3 = autocorr_feat(pd.DataFrame(sig), logger=logging) # 7 feature
        if id == 0:
            feat_data = pd.concat([X1, X2, X3], axis = 1).copy()
        else:
            feat_data = pd.concat([feat_data, pd.concat([X1, X2, X3], axis=1)], ignore_index=True)


    #  then concate with other features from matlab analysis.
    mat_feat_data = loadmat('..//feat_x.mat')
    columns = ['emav', 'ewl', 'fzc', 'asm', 'ass', 'msr', 'ltkeo', 'lcov', 'card',
    'ldasdv', 'ldamv', 'dvarv', 'vo', 'tm', 'damv',  'mad', 'iqr',  'cov',  'var', 'ae', 'iemg',
               'mav', 'ssc', 'zc', 'wl', 'rms', 'aac', 'dasdv', 'ld', 'mmav', 'mmav2', 'myop',
               'ssi', 'vare', 'wa', 'mfl','num_acc', 'num_dec', 'avg_dec_skew', 'avg_dec_kurt',
               'avg_dec_var', 'avg_dec_iqr', 'avg_dec_card', 'std_dec_skew', 'std_dec_kurt', 'std_dec_var',
            'std_dec_iqr', 'std_dec_card', 'num_pro_dec'] # from -matlab
    mat_feat = pd.DataFrame(data = mat_feat_data['feat'], columns= columns)

    all_feat = pd.concat([feat_data, mat_feat], axis=1)
    all_feat['label'] = my_y
    print(mat_feat.head(5))
    all_feat.to_csv('..//signalfeat.csv', index=False)
