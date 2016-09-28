import fix_mpl_svg

account_data_all_top100, org_data_all_top100 = zero_outside_top(account_data_all, org_data_all, 100)
num_to_choose = 5
account_data = np.copy(account_data_all_top100)
num_vids = account_data.shape[0]
num_hours = account_data.shape[1]

ix = np.argsort(account_data, 0)
days_since_published_top = np.zeros((num_to_choose, num_hours), dtype='float')
days_since_published_all = np.zeros((num_to_choose, num_hours), dtype='float')
for hour_ind in range(num_hours):
    for rank_ind in range(1, 2*num_to_choose+1):
        if rank_ind > num_to_choose:
            sample_rank_ind = np.random.randint(rank_ind, num_vids)
            while account_data[ix[-sample_rank_ind, hour_ind], hour_ind] == 0:
                sample_rank_ind = np.random.randint(rank_ind, num_vids)
        else:
            sample_rank_ind = rank_ind
        vid_ind_this_hour = ix[-sample_rank_ind, hour_ind]
        if pd.isnull(vid_dates[vid_ind_this_hour]):
            dt = np.nan
        else:
            dt = ((t_all[hour_ind] - vid_dates[vid_ind_this_hour]).total_seconds()/(60.*60.))
        if rank_ind > num_to_choose:
            days_since_published_all[rank_ind-num_to_choose-1, hour_ind] = dt
        else:
            days_since_published_top[rank_ind-1, hour_ind] = dt

dsp_top = days_since_published_top.ravel()
dsp_top = dsp_top[np.isnan(dsp_top)==False]
dsp_top_clipped = np.clip(dsp_top, 0., 12.)

dsp_all = days_since_published_all.ravel()
dsp_all = dsp_all[np.isnan(dsp_all)==False]
dsp_all_clipped = np.clip(dsp_all, 0., 12.)


fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_figwidth(16)
#ax.hist(dsp_all_clipped, np.linspace(0, 124, 125), normed=True, facecolor=[.1, .1, .1], alpha=.4, edgecolor='w', width=.9)
#ax.hist(dsp_top_clipped, np.linspace(0, 124, 125), normed=True, facecolor='r', alpha=.4, edgecolor='w', width=.9)
#ax.hist([dsp_top_clipped, dsp_all_clipped], np.linspace(0, 12, 13), normed=True, color=[[1, 0, 0], [.1, .1, .1]], alpha=.4, edgecolor='w', width=.4)
hist_out = ax.hist([dsp_top_clipped, dsp_all_clipped], np.linspace(0, 12, 13), normed=True, color=[[1, 0, 0], [.1, .1, .1]], alpha=.4, edgecolor='w', width=.4, cumulative=True)
ax.set_xlabel('Hours since published', fontname='FreeSans')
ax.set_ylabel('Cumulative % videos', fontname='FreeSans')

ax.spines['left'].set_bounds(0, 1)
ax.set_ylim((-.02, 1))
ax.set_yticks([0, .25, .5, .75, 1.])
ax.set_yticklabels([0, 25, 50, 75, 100])

ax.spines['bottom'].set_bounds(0, 12)
ax.set_xlim((-.7, 12.5))
xticks = np.linspace(0, 11, 12)

for bar_ind in range(len(hist_out[0][0])):
    ax.text(hist_out[1][bar_ind]+.3, hist_out[0][0][bar_ind]+.005, str(int(round(100.*hist_out[0][0][bar_ind])))+'%', horizontalalignment='center', color='r')
    ax.text(hist_out[1][bar_ind]+.8, hist_out[0][1][bar_ind]+.005, str(int(round(100.*hist_out[0][1][bar_ind])))+'%', horizontalalignment='center')

ax.set_xticks(xticks)

ax.tick_params(axis='both', which='major', pad=4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.text(1.2, .55, 'Top 5', color='r')
ax.text(12.2, .15, 'Top 100', color='k')

fig.savefig('time_since_published_hist.svg')
fix_mpl_svg.replace('time_since_published_hist.svg')


