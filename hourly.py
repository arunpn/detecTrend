import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import pandas as pd
import os
import h5py

def unique_preserve_order(A):
    """http://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved"""
    _temp, inds = np.unique(A, return_index=True)
    return A[np.sort(inds)]

DATA_PATH = '../data/'

date_strs = ['0905', '0906', '0907', '0908', '0909']

vid_list_file = h5py.File('../vid_list.hdf5', 'r')
all_vids = np.copy(vid_list_file['vids'])
is_test = np.copy(vid_list_file['is_test'])
vid_list_file.close()

all_current_viewers = np.zeros(((len(date_strs)+1)*60*24+1, len(all_vids)), dtype='int')

for vid_ind, vid in enumerate(all_vids):
    all_days_dfs = []
    for date_str_ind, date_str in enumerate(date_strs):
        file_name = DATA_PATH + vid + '.' + date_str + '.hdf5'
        in_df = pd.read_hdf(file_name)
        all_days_dfs.append(in_df)
        
    all_days_df = pd.concat(all_days_dfs)
    
    all_days_df = all_days_df.append(pd.DataFrame({'__time':pd.datetime(2016, 9, 5), 'account_id':'', 'org_id':''}, [-1]))
    all_days_df = all_days_df.append(pd.DataFrame({'__time':pd.datetime(2016, 9, 11), 'account_id':'', 'org_id':''}, [len(all_days_df.account_id)]))
    
    num_current_viewers = all_days_df.groupby('__time').account_id.nunique()
    mean_current_viewers = num_current_viewers.resample("1T").rolling(min_periods=1, window=1, center=True).sum()
    mean_current_viewers = mean_current_viewers.fillna(0)
    
    all_current_viewers[:, vid_ind] = mean_current_viewers

fig = plt.figure()
ax = fig.add_subplot(111)
#lower = np.percentile(all_current_viewers.astype('float'), 25., axis=1, interpolation='linear')
#upper = np.percentile(all_current_viewers.astype('float'), 75., axis=1, interpolation='linear')
lower = np.mean(all_current_viewers, axis=1) - np.std(all_current_viewers, axis=1)
upper = np.mean(all_current_viewers, axis=1) + np.std(all_current_viewers, axis=1)
ax.fill_between(np.arange(all_current_viewers.shape[0]), 0*lower, upper)
#ax.plot(np.median(all_current_viewers, axis=1))
ax.plot(np.mean(all_current_viewers, axis=1), 'b')
ax.set_xticks(np.arange(0, (len(date_strs)+1)*60*24+1, 60*24) - 700 + 60*24)
ax.set_xlim(700, 8000)
ax.set_xticklabels(date_strs)
fig.savefig('hourly.svg')
