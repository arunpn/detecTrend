from __future__ import print_function
import numpy as np
import pandas as pd
import h5py
from sklearn import linear_model, metrics
import matplotlib.pyplot as plt
plt.ion()
import os
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 24
#from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier

def get_vid_list(data_path):
    data_file_names = os.listdir(data_path)
    vids = list(set([dfn.split('.')[0] for dfn in data_file_names]))
    return vids

def get_vid_published_dates(vids):
    #published_date_df = pd.read_hdf('../vid_published_dates.hdf5')
    published_date_df = pd.read_hdf('../vid_published_datetimes.hdf5')
    vid_dates = []
    for vid in vids:
        if vid in published_date_df.vid.values:
            this_vid_d = published_date_df.datetime[published_date_df.vid==vid].values[0]
        else:
            print('no datetime for ' + vid)
            this_vid_d = pd.to_datetime('')
        vid_dates.append(this_vid_d)
    vid_dates = np.array(vid_dates)
    return vid_dates
    
def read_data(vids, data_path, t_division):
    
    this_data_file_name = t_division + 'data.hdf5'
    this_division_file_name = t_division + 'division.hdf5'
    data_path_file_names = os.listdir('../')
    if this_data_file_name in data_path_file_names and this_division_file_name in data_path_file_names:
        print('reading previously saved data')
        sum_hourly_viewers = pd.read_hdf('../' + this_division_file_name)
        in_file = h5py.File('../' + this_data_file_name, 'r')
        account_data = np.copy(in_file['account_data'])
        org_data = np.copy(in_file['org_data'])
        vids_out = np.copy(in_file['vids_out'])
        in_file.close()
    else:
        account_data = []
        org_data = []
        vids_out = []
    
        for vid_ind, vid in enumerate(vids):
            
            file_name = data_path + vid + '.06-17.08-02.hdf5'
            df = pd.read_hdf(file_name)
            if len(df) == 0:
                print(file_name, end='')
            else:
                df = df.append(pd.DataFrame({'first_start_time':pd.datetime(2016, 6, 16), 'account_id':'', 'org_id':''}, [-1]))
                df = df.append(pd.DataFrame({'first_start_time':pd.datetime(2016, 8, 3), 'account_id':'', 'org_id':''}, [len(df.account_id)]))
                
                df_reind = df.copy()
                df_reind = df_reind.set_index(['first_start_time'])
                
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
                vids_out.append(vid)
                print(str(vid_ind) + ' ', end='')
            
        account_data = np.array(account_data)
        org_data = np.array(org_data)
        vids_out = np.array(vids_out)

        print('\nsaving data to '+this_division_file_name+' '+this_data_file_name)
        sum_hourly_viewers.to_hdf('../' + this_division_file_name, 'w')
        out_file = h5py.File('../' + this_data_file_name, 'w')
        out_file.create_dataset('account_data', data=account_data)
        out_file.create_dataset('org_data', data=org_data)
        out_file.create_dataset('vids_out', data=vids_out)
        out_file.flush()
        out_file.close()

    t = sum_hourly_viewers.index
    return vids_out, t, account_data, org_data

def plot_peak_triggered(x_in, num_to_choose):
    most_popular_hourly = np.argsort(x_in, 0)[-num_to_choose:, :]
    most_popular = np.unique(most_popular_hourly)
    
    x = x_in[most_popular, :]
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
    
    ax.text(hour_range[hours_to_plot][-1]+.5, np.ma.mean(xx, 0)[hours_to_plot][-1]+.5, 'Mean', color='b')
    ax.text(hour_range[hours_to_plot][-1]+.5, np.ma.median(xx, 0)[hours_to_plot][-1]-.5, 'Median', color='k')

    ax.spines['left'].set_bounds(0, 30)
    ax.set_ylim((-2, 37))
    ax.set_yticks([0, 10, 20, 30])

    ax.spines['bottom'].set_bounds(-5, 10)
    ax.set_xlim((-6, 11))
    ax.set_xticks([-5, 0, 5, 10])
    ax.tick_params(axis='both', which='major', pad=4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    #fig.savefig('peak_triggered.svg')
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
    ax.plot(t, x.sum(0), color=[.7, .7, .7], zorder=10)
    ax.plot(t, sum_watching_top, color=[0, 0, 0], zorder=10)
    ax.axvspan(pd.Timestamp('2016-07-13').toordinal(), pd.Timestamp('2016-07-27').toordinal(), zorder=1, color=[.8, .8, 1.])
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Number of viewers', color='k')
    ax.set_yticks([0, 500, 1000, 1500])
    ax.spines['left'].set_bounds(0, 1500)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))
    
    ax2 = ax.twinx()
    ax2.plot(t, 100*sum_watching_top/x.sum(0), color='r')
    ax2.set_ylabel('Watching top ' + str(num_to_choose) + ' (%)', color='r')
    ax2.set_yticks([0, 25, 50, 75, 100])
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('r')
    
    ax.spines['bottom'].set_bounds(pd.Timestamp('2016-06-18').toordinal(), pd.Timestamp('2016-07-23').toordinal())
    ax2.spines['bottom'].set_bounds(pd.Timestamp('2016-06-18').toordinal(), pd.Timestamp('2016-07-23').toordinal())
    
    ax.set_xlim(pd.Timestamp('2016-06-17T12'), pd.Timestamp('2016-07-28'))
    ax.text(pd.Timestamp('2016-07-28').toordinal(), x.sum(0)[-1], 'Top 100', color=[.7, .7, .7])
    ax.text(pd.Timestamp('2016-07-28').toordinal(), sum_watching_top[-1], 'Top 5', color='k')
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=-30, ha='left')
    ax2.spines['bottom'].set_position(('outward', 10))
    
    #fig.savefig('time_series.svg')
    return fig

def extract_features(account_data, org_data, vid_dates, t_all, num_to_choose):
    norm_account_data = account_data/(account_data.sum(0)[np.newaxis, :])
    norm_org_data = org_data/(org_data.sum(0)[np.newaxis, :])
    num_vids = norm_account_data.shape[0]
    num_hours = norm_account_data.shape[1]
    x_tm1 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold viewers from time t-1
    x_tm2 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold viewers from time t-1
    dx_tm1 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold change of viewers between time t-1 and t-2
    xo_tm1 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold orgs from time t-1
    xt_tm1 = np.zeros((2*num_to_choose, num_hours), dtype='float') # will hold time from published from time t-1
    y_t = np.zeros((2*num_to_choose, num_hours), dtype='float')

    ix = np.argsort(norm_account_data, 0)

    for hour_ind in range(2, num_hours):
        for rank_ind in range(1, 2*num_to_choose+1):
            if rank_ind > num_to_choose:
                sample_rank_ind = np.random.randint(rank_ind, num_vids)
                while norm_account_data[ix[-sample_rank_ind, hour_ind], hour_ind-1] == 0:
                    sample_rank_ind = np.random.randint(rank_ind, num_vids)
            else:
                sample_rank_ind = rank_ind
            vid_ind_this_hour = ix[-sample_rank_ind, hour_ind]
            x_tm1[rank_ind-1, hour_ind] = norm_account_data[vid_ind_this_hour, hour_ind-1]
            x_tm2[rank_ind-1, hour_ind] = norm_account_data[vid_ind_this_hour, hour_ind-2]
            dx_tm1[rank_ind-1, hour_ind] = norm_account_data[vid_ind_this_hour, hour_ind-1] - norm_account_data[vid_ind_this_hour, hour_ind-2]
            xo_tm1[rank_ind-1, hour_ind] = norm_org_data[vid_ind_this_hour, hour_ind-1]
            if pd.isnull(vid_dates[vid_ind_this_hour]):
                dt = np.nan
            else:
                dt = (t_all[hour_ind-1] - vid_dates[vid_ind_this_hour]).total_seconds()/(60.*60.)
            xt_tm1[rank_ind-1, hour_ind] = dt
            y_t[rank_ind-1, hour_ind] = norm_account_data[vid_ind_this_hour, hour_ind]
        
    x_tm1 = x_tm1[:, 2:]
    x_tm2 = x_tm2[:, 2:]
    dx_tm1 = dx_tm1[:, 2:]
    xo_tm1 = xo_tm1[:, 2:]
    xt_tm1 = xt_tm1[:, 2:]
    y_t = y_t[:, 2:]

    x_tm1 = x_tm1.reshape(2*num_to_choose*(num_hours - 2), 1)
    x_tm2 = x_tm2.reshape(2*num_to_choose*(num_hours - 2), 1)
    dx_tm1 = dx_tm1.reshape(2*num_to_choose*(num_hours - 2), 1)
    xo_tm1 = xo_tm1.reshape(2*num_to_choose*(num_hours - 2), 1)
    xt_tm1 = xt_tm1.reshape(2*num_to_choose*(num_hours - 2), 1)
    xt_tm1[np.isnan(xt_tm1)] = np.nanmean(xt_tm1)
    y_t = y_t.reshape(2*num_to_choose*(num_hours - 2), 1)

    x_out = np.concatenate((x_tm1, dx_tm1, xo_tm1, xt_tm1), axis=1)

    #x_out = x_tm1
    #x_out = x_tm2
    #x_out = xo_tm1
    #x_out = dx_tm1
    #x_out = xt_tm1

    y_out = y_t
    
    return x_out, y_out
    
def viewers2score(y, num_hours, num_to_choose):
    """example:
    y = np.random.random_integers(0, 5, (6, 10))
    y_score = viewers2score(y.ravel(), y.shape[1] + 2, y.shape[0]/2)
    """
    hourly_y = y.reshape(2*num_to_choose, num_hours - 2)
    score = np.zeros_like(hourly_y)

    for hour_ind in range(num_hours - 2):
        h = hourly_y[:, hour_ind]
        uh = np.sort(np.unique(h)).tolist()[::-1]
        sc = num_to_choose - np.array([uh.index(hh) for hh in h])
        sc = np.clip(sc, 0, num_to_choose)
        while np.sum(sc > 0) > num_to_choose:
            # randomly eliminate low scores
            ind = np.argmax(sc == sc[sc>0].min())
            sc[ind] = 0
        
        score[:, hour_ind] = sc
    
    return score

def zero_outside_top(account_data, org_data, full_set_size):
    ix = np.argsort(account_data, 0)
    account_data_out = np.copy(account_data)
    org_data_out = np.copy(org_data)
    for hour_ind in range(account_data.shape[1]):
        thesh_this_hour = account_data_out[ix[-full_set_size, hour_ind], hour_ind]
        below_thesh_inds = account_data_out[:, hour_ind] < thesh_this_hour
        account_data_out[below_thesh_inds, hour_ind] = 0.
        org_data_out[below_thesh_inds, hour_ind] = 0.
        
        while np.sum(account_data_out[:, hour_ind] > 0) > full_set_size:
            # randomly eliminate lower viewership videos
            ind = np.argmax(account_data_out[:, hour_ind] == thesh_this_hour)
            account_data_out[ind, hour_ind] = 0
            org_data_out[ind, hour_ind] = 0
    return account_data_out, org_data_out

DATA_PATH = '../data/'
num_to_choose = 5

t_division = "h"

all_vids = get_vid_list(DATA_PATH)
vids_out, t_all, account_data_all, org_data_all = read_data(all_vids, DATA_PATH, t_division)

#fig_peak_triggered = plot_peak_triggered(account_data_all, num_to_choose)
#sum_watching_top = calc_sum_watching_top(account_data_all, num_to_choose)
#account_data_all_top100, org_data_all_top100 = zero_outside_top(account_data_all, org_data_all, 100)
#fig_time_series = plot_time_series(t_all, account_data_all_top100, sum_watching_top, num_to_choose)

vid_dates = get_vid_published_dates(vids_out)

num_hours_all = account_data_all.shape[1]

for data_set in ['train', 'test']:
    if data_set == 'train':
        is_in_set_hour = np.arange(num_hours_all) < num_hours_all - 2*7*24
    elif data_set == 'test':
        is_in_set_hour = np.arange(num_hours_all) >= num_hours_all - 2*7*24

    num_hours_set = is_in_set_hour.sum()

    account_data = np.copy(account_data_all[:, is_in_set_hour])
    org_data = np.copy(org_data_all[:, is_in_set_hour])
    x, y = extract_features(account_data, org_data, vid_dates, t_all, num_to_choose)
    
    y_score = viewers2score(y, num_hours_set, num_to_choose) > 0
    
    if data_set == 'train':
        figROC = plt.figure()
        axROC = figROC.add_subplot(111)
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        
        figRaw = plt.figure()
        axRaw = figRaw.add_subplot(111)

    
    print('Coefficients: ', regr.coef_)
    print('Variance score '+data_set+' set: %.2f' % regr.score(x, y))
    
    predicted_y = regr.predict(x)
    predicted_y[x[:, 0]==0] = 0 # otherwise unfair selection bias
    
    #predicted_y = x
    predicted_y_score = viewers2score(predicted_y, num_hours_set, num_to_choose) > 0
    
    print('LR Accuracy score '+data_set+' set: %.2f' % metrics.accuracy_score(y_score.ravel(), predicted_y_score.ravel()))
    
    f, t, th = metrics.roc_curve(y_score.ravel(), predicted_y)
    axROC.plot(f, t, label=data_set+'Linear regression')
    
    axRaw.plot(x[y_score.ravel()==0, 0].ravel(), y[y_score.ravel()==0].ravel(), '.', label=data_set+' True negatives')
    axRaw.plot(x[y_score.ravel()>0, 0].ravel(), y[y_score.ravel()>0].ravel(), '.', label=data_set+' True positives')
    #if data_set == 'test':
    #    axRaw.plot(x[y_score.ravel()==0, 0].ravel(), x[y_score.ravel()==0, 1].ravel(), '.k', label=data_set+' True negatives')
    #    axRaw.plot(x[y_score.ravel()>0, 0].ravel(), x[y_score.ravel()>0, 1].ravel(), '.r', label=data_set+' True positives')
    
    if data_set == 'train':
        #clf = svm.SVC()
        #clf = tree.DecisionTreeClassifier()
        clf = RandomForestClassifier(max_depth=4)
        clf.fit(x, y_score.ravel())
        
    #predicted_y = clf.decision_function(x)
    predicted_y = clf.predict_proba(x)[:, 1]
    predicted_y[x[:, 0]==0] = 0 # otherwise unfair selection bias
    predicted_y_score = viewers2score(predicted_y, num_hours_set, num_to_choose) > 0
    print('RF Accuracy score '+data_set+' set: %.2f' % metrics.accuracy_score(y_score.ravel(), predicted_y_score.ravel()))

    f, t, th = metrics.roc_curve(y_score.ravel(), predicted_y.ravel())
    axROC.plot(f, t, ':', label=data_set+' Random forest')
    
    axRaw.plot(x[:, 0], predicted_y, '.', label=data_set+' Random forest')
    
    

axROC.legend(loc='lower right')
axRaw.legend(loc='lower right')

