from __future__ import print_function
import numpy as np
import pandas as pd
import h5py
from sklearn import linear_model
import matplotlib.pyplot as plt
plt.ion()

def extract_features(vids):

    DATA_PATH = '../data/'
    x_date_strs = ['0905', '0906', '0907', '0908']
    y_date_str = '0909'

    x = np.zeros((len(vids), len(x_date_strs)), dtype='int') - 1
    y = np.zeros((len(vids), 1), dtype='int') - 1

    for vid_ind, vid in enumerate(vids):
        
        for x_date_str_ind, x_date_str in enumerate(x_date_strs):
            x_file_name = DATA_PATH + vid + '.' + x_date_str + '.hdf5'
            x_df = pd.read_hdf(x_file_name)
            account_array_this_video = x_df['account_id'].values
            num_unique_accounts = len(np.unique(account_array_this_video))
        
            x[vid_ind, x_date_str_ind] = num_unique_accounts
        
        y_file_name = DATA_PATH + vid + '.' + y_date_str + '.hdf5'
        y_df = pd.read_hdf(y_file_name)
        account_array_this_video = y_df['account_id'].values
        num_unique_accounts = len(np.unique(account_array_this_video))
        
        y[vid_ind, 0] = num_unique_accounts
        print(str(vid_ind) + ' ', end='')
    print('\n')
    return x, y

vid_list_file = h5py.File('../vid_list.hdf5', 'r')
all_vids = np.copy(vid_list_file['vids'])
is_test = np.copy(vid_list_file['is_test'])
vid_list_file.close()

train_vids = all_vids[is_test==False]

x_train, y_train = extract_features(train_vids)

x_train_mean, x_train_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
y_train_mean, y_train_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)

x_train_z = (x_train - x_train_mean)/x_train_std
y_train_z = (y_train - y_train_mean)/y_train_std

regr = linear_model.LinearRegression()
regr.fit(x_train_z, y_train_z)

fig = plt.figure()
ax_train0 = fig.add_subplot(2, 2, 1)
ax_train0.plot(x_train_z[:, 0], y_train_z, '.')
ax_train0.plot(x_train_z[:, 0], regr.predict(x_train_z), 'r.')
ax_train1 = fig.add_subplot(2, 2, 2)
ax_train1.plot(x_train_z[:, 1], y_train_z, '.')
ax_train1.plot(x_train_z[:, 1], regr.predict(x_train_z), 'r.')

print('Coefficients: ', regr.coef_)

######################################## test #########################

test_vids = all_vids[is_test==True]

x_test, y_test = extract_features(test_vids)

x_test_z = (x_test - x_train_mean)/x_train_std
y_test_z = (y_test - y_train_mean)/y_train_std

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(x_test_z) - y_test_z) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test_z, y_test_z))

ax_test0 = fig.add_subplot(2, 2, 3)
ax_test0.plot(x_test_z[:, 0], y_test_z, '.')
ax_test0.plot(x_test_z[:, 0], regr.predict(x_test_z), 'r.')
ax_test1 = fig.add_subplot(2, 2, 4)
ax_test1.plot(x_test_z[:, 1], y_test_z, '.')
ax_test1.plot(x_test_z[:, 1], regr.predict(x_test_z), 'r.')

figROC = plt.figure()
axROC = figROC.add_subplot(111)
f, t, th = sklearn.metrics.roc_curve(y_test_z>0, regr.predict(x_test_z))
axROC.plot(f, t)
