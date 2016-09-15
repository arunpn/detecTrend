from __future__ import print_function
import numpy as np
import pandas as pd
import h5py
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.ion()

DATA_PATH = '../current/data/'

vid_list_file = h5py.File('../current/vid_list.hdf5', 'r')
vids = np.copy(vid_list_file['vids'])
is_test = np.copy(vid_list_file['is_test'])
vid_list_file.close()


train_vids = vids[is_test==False]

x_date_str = '0908'
y_date_str = '0909'

x_train = np.zeros((len(train_vids), 1), dtype='int') - 1
y_train = np.zeros((len(train_vids), 1), dtype='int') - 1

for train_vid_ind, train_vid in enumerate(train_vids):
    x_file_name = DATA_PATH + train_vid + '.' + x_date_str + '.hdf5'
    x_df = pd.read_hdf(x_file_name)
    account_array_this_video = x_df['account_id'].values
    num_unique_accounts = len(np.unique(account_array_this_video))
    
    x_train[train_vid_ind, 0] = num_unique_accounts
    
    y_file_name = DATA_PATH + train_vid + '.' + y_date_str + '.hdf5'
    y_df = pd.read_hdf(y_file_name)
    account_array_this_video = y_df['account_id'].values
    num_unique_accounts = len(np.unique(account_array_this_video))
    
    y_train[train_vid_ind, 0] = num_unique_accounts
    print(train_vid_ind, end=' ')

x_train_mean, x_train_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
y_train_mean, y_train_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)

x_train_z = (x_train - x_train_mean)/x_train_std
y_train_z = (y_train - y_train_mean)/y_train_std

regr = linear_model.LinearRegression()
regr.fit(x_train_z, y_train_z)

fig = plt.figure()
ax_train = fig.add_subplot(2, 1, 1)
ax_train.plot(x_train_z, y_train_z, '.')
ax_train.plot(x_train_z, regr.predict(x_train_z), color='blue', linewidth=3)

print('\nCoefficients: ', regr.coef_)

######################################## test #########################
test_vids = vids[is_test==True]
x_test = np.zeros((len(test_vids), 1), dtype='int') - 1
y_test = np.zeros((len(test_vids), 1), dtype='int') - 1

for test_vid_ind, test_vid in enumerate(test_vids):
    x_file_name = DATA_PATH + test_vid + '.' + x_date_str + '.hdf5'
    x_df = pd.read_hdf(x_file_name)
    account_array_this_video = x_df['account_id'].values
    num_unique_accounts = len(np.unique(account_array_this_video))
    
    x_test[test_vid_ind, 0] = num_unique_accounts
    
    y_file_name = DATA_PATH + test_vid + '.' + y_date_str + '.hdf5'
    y_df = pd.read_hdf(y_file_name)
    account_array_this_video = y_df['account_id'].values
    num_unique_accounts = len(np.unique(account_array_this_video))
    
    y_test[test_vid_ind, 0] = num_unique_accounts
    print(test_vid_ind, end=' ')

x_test_z = (x_test - x_train_mean)/x_train_std
y_test_z = (y_test - y_train_mean)/y_train_std

# The mean square error
print("\nResidual sum of squares: %.2f"
      % np.mean((regr.predict(x_test_z) - y_test_z) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test_z, y_test_z))

ax_test = fig.add_subplot(2, 1, 2)
ax_test.plot(x_test_z, y_test_z, '.')
ax_test.plot(x_test_z, regr.predict(x_test_z), color='blue', linewidth=3)

