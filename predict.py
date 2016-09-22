from __future__ import print_function
import numpy as np
import pandas as pd
import h5py
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt
plt.ion()
import os

def get_vid_list(data_path):
    data_file_names = os.listdir(data_path)
    vids = list(set([dfn.split('.')[0] for dfn in data_file_names]))
    return vids

def read_data(vids, data_path, t_division):
    
    account_data = []
    org_data = []
    
    this_data_file_name = t_division + 'data.hdf5'
    this_division_file_name = t_division + 'division.hdf5'
    data_path_file_names = os.listdir('../')
    if this_data_file_name in data_path_file_names and this_division_file_name in data_path_file_names:
        print('reading previously saved data')
        sum_hourly_viewers = pd.read_hdf('../' + this_division_file_name)
        in_file = h5py.File('../' + this_data_file_name, 'r')
        account_data = np.copy(in_file['account_data'])
        org_data = np.copy(in_file['org_data'])
        in_file.close()
    else:
        for vid_ind, vid in enumerate(vids):
            
            file_name = data_path + vid + '.06-17.08-02.hdf5'
            df = pd.read_hdf(file_name)
            if len(df) == 0:
                print(file_name, end='')
            else:
                df = df.append(pd.DataFrame({'submission_time':pd.datetime(2016, 6, 16), 'account_id':'', 'org_id':''}, [-1]))
                df = df.append(pd.DataFrame({'submission_time':pd.datetime(2016, 8, 3), 'account_id':'', 'org_id':''}, [len(df.account_id)]))
                
                df_reind = df.copy()
                df_reind = df_reind.set_index(['submission_time'])
                
                year_month_day_hour = pd.to_datetime(2016*1000000 + df_reind.index.month*10000 + df_reind.index.day*100 + df_reind.index.hour, format='%Y%m%d%H')
                df_reind['year_month_day_hour'] = year_month_day_hour
                num_current_viewers = df_reind.groupby('year_month_day_hour').account_id.nunique()
                num_current_orgs = df_reind.groupby('year_month_day_hour').org_id.nunique()
                
                sum_hourly_viewers = num_current_viewers.resample(t_division).sum()
                sum_hourly_viewers = sum_hourly_viewers.fillna(0)
                sum_hourly_viewers = sum_hourly_viewers['2016-06-18 00:00:00': '2016-07-27 0:00:00']
                
                sum_hourly_orgs = num_current_orgs.resample(t_division).sum()
                sum_hourly_orgs = sum_hourly_orgs.fillna(0)
                sum_hourly_orgs = sum_hourly_orgs['2016-06-18 00:00:00': '2016-07-27 0:00:00']
                
                shu_array = sum_hourly_viewers.values.astype('float')
                sho_array = sum_hourly_orgs.values.astype('float')
                
                account_data.append(shu_array.tolist())
                org_data.append(sho_array.tolist())
                print(str(vid_ind) + ' ', end='')
            
        account_data = np.array(account_data)
        org_data = np.array(org_data)

        print('\nsaving data to '+this_division_file_name+' '+this_data_file_name)
        sum_hourly_viewers.to_hdf('../' + this_division_file_name, 'w')
        out_file = h5py.File('../' + this_data_file_name, 'w')
        out_file.create_dataset('account_data', data=account_data)
        out_file.create_dataset('org_data', data=org_data)
        out_file.flush()
        out_file.close()

    t = sum_hourly_viewers.index
    return t, account_data, org_data

def plot_peak_triggered(x):
    mask_x = np.ma.masked_all_like(x)
    mask_x.fill(np.nan)
    padded_x = np.ma.concatenate((mask_x, x, mask_x), axis=1)
    maxinds = np.argmax(x, axis=1)
    xx = np.ma.masked_all((x.shape[0], x.shape[1]*2), dtype='float')
    for vi, maxind in enumerate(maxinds):
        #if maxind >=2 and maxind <= x.shape[1]-1:
        #xx[vi, :] = x[vi, maxind-2: maxind+3]
        #if len(x[vi, maxind-2: maxind+3]) > 0 and np.nanmax(x[vi, maxind-2: maxind+3])>2000:
        xx[vi, :] = padded_x[vi, maxind: maxind+2*x.shape[1]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(-5, 10)
    upper = np.nanpercentile(xx, 75, axis=0)
    lower = np.nanpercentile(xx, 25, axis=0)
    hour_range = np.arange(xx.shape[1]) - xx.shape[1]/2.
    hours_to_plot = (hour_range >=-5) & (hour_range <= 10)
    ax.plot(hour_range[hours_to_plot], np.ma.mean(xx, 0)[hours_to_plot], '.-b')
    ax.plot(hour_range[hours_to_plot], np.ma.median(xx, 0)[hours_to_plot], '.-k')
    #ax.plot(np.arange(xx.shape[1]) - xx.shape[1]/2., np.nanmedian(xx, 0), '.-')
    ax.fill_between(hour_range[hours_to_plot], upper[hours_to_plot], lower[hours_to_plot])

    ax.set_xlabel('Hours after peak', fontname='FreeSans')
    ax.set_ylabel('Number of viewers', fontname='FreeSans')
    
    ax.text(hour_range[hours_to_plot][-1]+.5, np.ma.mean(xx, 0)[hours_to_plot][-1], 'Mean', color='b')
    ax.text(hour_range[hours_to_plot][-1]+.5, np.ma.median(xx, 0)[hours_to_plot][-1], 'Median', color='k')

    ax.spines['left'].set_bounds(0, 300)
    ax.set_ylim((-20, 350))
    ax.set_yticks([0, 100, 200, 300])

    ax.spines['bottom'].set_bounds(-5, 10)
    ax.set_xlim((-6, 11))
    ax.set_xticks([-5, 0, 5, 10])
    ax.tick_params(axis='both', which='major', pad=4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    fig.savefig('peak_triggered.svg')
    return fig

def calc_sum_watching_top(data, num_to_choose):
    ix = np.argsort(data, 0)
    top_each_hour = ix[-num_to_choose:, :]
    sum_watching_top = np.zeros(data.shape[1], dtype='float')
    for hour_ind in range(data.shape[1]):
        sum_watching_top[hour_ind] = np.sum(data[ix[-num_to_choose:, hour_ind], hour_ind])
    return sum_watching_top

def plot_time_series(t, x, sum_watching_top, num_to_choose):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, x.sum(0), color=[.8, .8, .8])
    ax.plot(t, sum_watching_top, color=[0, 0, 0])
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Number of viewers', color='k')

    ax2 = ax.twinx()
    ax2.plot(t, 100*sum_watching_top/x.sum(0), color='r')
    ax2.set_ylabel('Watching top ' + str(num_to_choose) + ' (%)', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ax2.set_ylim((0, 100))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('r')

    ax.set_xlim(pd.Timestamp('2016-06-14'), pd.Timestamp('2016-07-28'))
    ax.text(pd.Timestamp('2016-06-17'), x.sum(0)[0]+2000, 'All', color=[.8, .8, .8])
    ax.text(pd.Timestamp('2016-06-14'), sum_watching_top[0]-500, 'Top 5', color='k')
    fig.savefig('time_series.svg')
    return fig

def extract_features(account_data, org_data, num_to_choose):
    norm_account_data = account_data/(account_data.sum(0)[np.newaxis, :])
    norm_org_data = org_data/(org_data.sum(0)[np.newaxis, :])
    num_hours = norm_account_data.shape[1]
    x_tm1 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold viewers from time t-1
    dx_tm1 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold change of viewers between time t-1 and t-2
    xo_tm1 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold orgs from time t-1
    y_t = np.zeros((2*num_to_choose, num_hours), dtype='float')

    ix = np.argsort(norm_account_data, 0)

    for hour_ind in range(2, num_hours):
        for rank_ind in range(1, 2*num_to_choose+1):
            vid_ind_this_hour = ix[-rank_ind, hour_ind]
            x_tm1[rank_ind-1, hour_ind] = norm_account_data[vid_ind_this_hour, hour_ind-1]
            dx_tm1[rank_ind-1, hour_ind] = norm_account_data[vid_ind_this_hour, hour_ind-1] - norm_account_data[vid_ind_this_hour, hour_ind-2]
            xo_tm1[rank_ind-1, hour_ind] = norm_org_data[vid_ind_this_hour, hour_ind-1]
            y_t[rank_ind-1, hour_ind] = norm_account_data[vid_ind_this_hour, hour_ind]
        
    x_tm1 = x_tm1[:, 2:]
    dx_tm1 = dx_tm1[:, 2:]
    xo_tm1 = xo_tm1[:, 2:]
    y_t = y_t[:, 2:]

    x_tm1 = x_tm1.reshape(2*num_to_choose*(num_hours - 2), 1)
    dx_tm1 = dx_tm1.reshape(2*num_to_choose*(num_hours - 2), 1)
    xo_tm1 = xo_tm1.reshape(2*num_to_choose*(num_hours - 2), 1)
    y_t = y_t.reshape(2*num_to_choose*(num_hours - 2), 1)

    #x_out = np.concatenate((x_tm1, dx_tm1), axis=1)
    x_out = np.concatenate((x_tm1, dx_tm1, xo_tm1), axis=1)

    y_out = y_t
    
    return x_out, y_out
    
def viewers2score(y, num_hours, num_to_choose):
    hourly_y = y.reshape(2*num_to_choose, num_hours - 2)
    score = np.zeros_like(hourly_y)

    for hour_ind in range(num_hours - 2):
        h = hourly_y[:, hour_ind]
        uh = np.sort(np.unique(h)).tolist()[::-1]
        sc = num_to_choose - np.array([uh.index(hh) for hh in h])
        sc = np.clip(sc, 0, num_to_choose)
        score[:, hour_ind] = sc
    
    return score

DATA_PATH = '../data/'
num_to_choose = 10

t_division = "h"

all_vids = get_vid_list(DATA_PATH)
t_all, account_data_all, org_data_all = read_data(all_vids, DATA_PATH, t_division)
fig_peak_triggered = plot_peak_triggered(account_data_all)
sum_watching_top = calc_sum_watching_top(account_data_all, num_to_choose)
fig_time_series = plot_time_series(t_all, account_data_all, sum_watching_top, num_to_choose)

num_hours_all = account_data_all.shape[1]

is_test_hour = np.arange(num_hours_all) >= num_hours_all - 7*24
is_train_hour = np.arange(num_hours_all) < num_hours_all - 7*24

num_hours_test = is_test_hour.sum()
num_hours_train = is_train_hour.sum()

account_data_train = np.copy(account_data_all[:, is_train_hour])
org_data_train = np.copy(org_data_all[:, is_train_hour])
x_train, y_train = extract_features(account_data_train, org_data_train, num_to_choose)

x_train_mean, x_train_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
y_train_mean, y_train_std = np.mean(y_train, axis=0), np.std(y_train, axis=0)
x_train_z = x_train
y_train_z = y_train
#x_train_z = (x_train - x_train_mean)/x_train_std
#y_train_z = (y_train - y_train_mean)/y_train_std

regr = linear_model.LinearRegression()
#regr.fit_intercept = False
regr.fit(x_train_z, y_train_z)

print('Coefficients: ', regr.coef_)
print('Variance score training set: %.2f' % regr.score(x_train_z, y_train_z))

predicted_y_train_z = regr.predict(x_train_z)
predicted_y_train = predicted_y_train_z
#predicted_y_train = predicted_y_train_z*y_train_std + y_train_mean

fig_train = plt.figure()
ax_train0 = fig_train.add_subplot(2, 1, 1)
ax_train0.plot(y_train, predicted_y_train, 'b.')
#ax_train0.plot((0, 2300), (0, 2300), 'k:')
ax_train1 = fig_train.add_subplot(2, 1, 2)
ax_train1.plot(np.sqrt(y_train), np.sqrt(predicted_y_train), 'b.')
#ax_train1.plot((0, np.sqrt(2300)), (0, np.sqrt(2300)), 'k:')

score_train = viewers2score(y_train, num_hours_train, num_to_choose)
predicted_score_train = viewers2score(predicted_y_train, num_hours_train, num_to_choose)

hourly_rmse = np.sqrt(np.mean((score_train[:num_to_choose, :] - predicted_score_train[:num_to_choose, :])**2, axis=0))
hourly_sae = np.sum(np.abs(score_train[:num_to_choose, :] - predicted_score_train[:num_to_choose, :]), axis=0)/float(sum(range(num_to_choose+1)))
hourly_sae = np.sum(np.abs(score_train), axis=0)
######################################## test #########################

account_data_test = np.copy(account_data_all[:, is_test_hour])
org_data_test = np.copy(org_data_all[:, is_test_hour])
x_test, y_test = extract_features(account_data_test, org_data_test, num_to_choose)

x_test_z = x_test
y_test_z = y_test
#x_test_z = (x_test - x_train_mean)/x_train_std
#y_test_z = (y_test - y_train_mean)/y_train_std

# Explained variance score: 1 is perfect prediction
print('Variance score test set: %.2f' % regr.score(x_test_z, y_test_z))
    
predicted_y_test_z = regr.predict(x_test_z)
predicted_y_test = predicted_y_test_z
#predicted_y_test = predicted_y_test_z*y_train_std + y_train_mean

fig_test = plt.figure()
ax_test0 = fig_test.add_subplot(2, 1, 1)
ax_test0.plot(y_test, predicted_y_test, 'b.')
#ax_test0.plot((0, 2300), (0, 2300), 'k:')
ax_test1 = fig_test.add_subplot(2, 1, 2)
ax_test1.plot(np.sqrt(y_test), np.sqrt(predicted_y_test), 'b.')
#ax_test1.plot((0, np.sqrt(2300)), (0, np.sqrt(2300)), 'k:')

figROC = plt.figure()
axROC = figROC.add_subplot(111)
f, t, th = metrics.roc_curve(y_test_z>0, regr.predict(x_test_z))
axROC.plot(f, t)

score_test = viewers2score(y_test, num_hours_test, num_to_choose)
predicted_score_test = viewers2score(predicted_y_test, num_hours_test, num_to_choose)
hourly_sae = np.sum(np.abs(score_test[:num_to_choose, :] - predicted_score_test[:num_to_choose, :]), axis=0)/float(sum(range(num_to_choose+1)))

plt.figure()
plt.hist(hourly_sae, num_to_choose)
