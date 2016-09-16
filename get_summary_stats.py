from __future__ import print_function
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
plt.ion()

DATA_PATH = '../data/'

vid_list_file = h5py.File('../vid_list.hdf5', 'r')
all_vids = np.copy(vid_list_file['vids'])
is_test = np.copy(vid_list_file['is_test'])
vid_list_file.close()

x_date_strs = ['0905','0906', '0907', '0908']
y_date_str = '0909'
dates_to_compare = x_date_strs + [y_date_str]

accounts_set = set()
orgs_set = set()
num_unique_accounts = np.zeros((len(all_vids), len(dates_to_compare)))

for vid_ind, vid in enumerate(all_vids):
    for date_str_ind, date_str in enumerate(dates_to_compare):
        file_name = DATA_PATH + vid + '.' + date_str + '.hdf5'
        df = pd.read_hdf(file_name)
        
        accounts_set_this_day = set(df['account_id'].values)
        org_set_this_day = set(df['org_id'].values)
        
        accounts_set.update(accounts_set_this_day)
        orgs_set.update(org_set_this_day)
        
        num_unique_accounts_this_day = len(accounts_set_this_day)
        num_unique_accounts[vid_ind, date_str_ind] = num_unique_accounts_this_day

    print(str(vid_ind) + ' ', end='')

print('\n Unique accounts: ' + str(len(accounts_set)))
print('Unique orgs: ' + str(len(orgs_set)))

least_popular_ind = np.argmin(num_unique_accounts.sum(1))
most_popular_ind = np.argmax(num_unique_accounts.sum(1))

fig_hist = plt.figure()
ax_hist = fig_hist.add_subplot(111)
ax_hist.hist(np.clip(num_unique_accounts.sum(1), 0, 2000), 50)
#bins = np.arange(0, 2001, 50)
#bins[-1] = 20000
#cnts, bins = np.histogram(num_unique_accounts.sum(1), bins)
#ax_hist.plot(bins[:-1]+25, cnts)


