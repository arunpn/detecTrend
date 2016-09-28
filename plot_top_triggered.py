def plot_top_triggered(t_all, x_in, num_to_choose, vids_out=None, plot_typical=False):
    #x_in = np.copy(account_data_all)
    #plot_typical = True

    most_popular_hourly = np.argsort(x_in, 0)[-num_to_choose:, :]
    most_popular = np.unique(most_popular_hourly)
    x = x_in[most_popular, :]
    v = vids_out[most_popular]
    num_hours = x.shape[1]
    score = np.zeros_like(x)

    for hour_ind in range(num_hours):
        h = x[:, hour_ind]
        uh = np.sort(np.unique(h)).tolist()[::-1]
        sc = num_to_choose - np.array([uh.index(hh) for hh in h])
        sc = np.clip(sc, 0, num_to_choose)
        while np.sum(sc > 0) > num_to_choose:
            # randomly eliminate low scores
            ind = np.argmax(sc == sc[sc>0].min())
            sc[ind] = 0
        
        score[:, hour_ind] = sc
    
    mask_x = np.ma.masked_all_like(x)
    mask_x.fill(np.nan)
    padded_x = np.ma.concatenate((mask_x, x, mask_x), axis=1)
    #maxinds = np.argmax(x, axis=1)
    maxinds = np.argmax(score, axis=1)
    xx = np.ma.masked_all((x.shape[0], x.shape[1]*2), dtype='float')
    for vi, maxind in enumerate(maxinds):
        xx[vi, :] = padded_x[vi, maxind: maxind+2*x.shape[1]]

    fig = plt.figure()  
    ax = fig.add_subplot(111)
    hour_range = np.arange(xx.shape[1]) - xx.shape[1]/2.
    hours_to_plot = (hour_range >=-5) & (hour_range <= 10)

    if plot_typical:
        #find_typical:
        published_date_df = pd.read_hdf('../vid_published_datetimes.hdf5')
        typical_vi = np.argmin(np.sum((xx[:, hours_to_plot] - np.ma.median(xx, 0)[hours_to_plot])**2, axis=1))
        tt = t_all[maxinds[typical_vi]-5: maxinds[typical_vi]+11]
        print('https://www.youtube.com/watch?v='+v[typical_vi])
        print(published_date_df[published_date_df.vid=='rwEzKUp19vE'])

        ax.plot(hour_range[hours_to_plot], xx[typical_vi, hours_to_plot], '.-b')
        ax.set_xlabel('Date', fontname='FreeSans')
    else:
        upper = np.nanpercentile(xx, 75, axis=0)
        lower = np.nanpercentile(xx, 25, axis=0)
        #ax.plot(hour_range[hours_to_plot], np.ma.mean(xx, 0)[hours_to_plot], '.-b')
        ax.plot(hour_range[hours_to_plot], np.ma.median(xx, 0)[hours_to_plot], '.-k')
        #ax.plot(np.arange(xx.shape[1]) - xx.shape[1]/2., np.nanmedian(xx, 0), '.-')
        ax.fill_between(hour_range[hours_to_plot], upper[hours_to_plot], lower[hours_to_plot])

        ax.set_xlabel('Hours after reaching top 5', fontname='FreeSans')
        
        #ax.text(hour_range[hours_to_plot][-1]+.5, np.ma.mean(xx, 0)[hours_to_plot][-1]+.5, 'Mean', color='b')
        ax.text(hour_range[hours_to_plot][-1]+.5, np.ma.median(xx, 0)[hours_to_plot][-1]-.5, 'Median', color='k')
        ax.text(hour_range[hours_to_plot][-1]+.5, upper[hours_to_plot][-1], '75%', color=[.6, .6, .6])
        ax.text(hour_range[hours_to_plot][-1]+.5, lower[hours_to_plot][-1]-2, '25%', color=[.6, .6, .6])
    
    ax.set_ylabel('Number of viewers', fontname='FreeSans')
    ax.spines['left'].set_bounds(0, 30)
    ax.set_ylim((-2, 37))
    ax.set_yticks([0, 10, 20, 30])
    
    ax.spines['bottom'].set_bounds(-5, 10)
    ax.set_xlim((-6, 11))
    ax.set_xticks([-5, 0, 5, 10])
    if plot_typical:
        xticklabels = [str(tt[t_ind].month) + '/' + str(tt[t_ind].day) + ' ' + str(np.mod(tt[t_ind].hour, 12)) + ['AM', 'PM'][int(tt[t_ind].hour>12)] for t_ind in [0, 5, 10, 15]]
        ax.set_xticklabels(xticklabels)
        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=-30, ha='left')
    
    ax.tick_params(axis='both', which='major', pad=4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if plot_typical:
        file_name = 'typical_top_triggered.svg'
    else:
        file_name = 'top_triggered.svg'
    fig.savefig(file_name)
    return fig




