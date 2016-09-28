import fix_mpl_svg

def plot_performance_hist(predicted_y_score, y_score, num_to_choose):

    hourly_num_right = num_to_choose - np.sum(np.abs(predicted_y_score[:num_to_choose, :].astype('float') - y_score[:num_to_choose, :].astype('float')), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    hist_out = ax.hist(hourly_num_right, np.linspace(-.5, 5.5, 7), normed=True, facecolor='r', alpha=.4, edgecolor='w', width=.9)

    ax.set_xlabel('Number correct out of top 5', fontname='FreeSans')
    ax.set_ylabel('% Test set', fontname='FreeSans')

    ax.spines['left'].set_bounds(0, .5)
    ax.set_ylim((-.02, .5))
    ax.set_yticks([0, .25, .5])
    ax.set_yticklabels([0, 25, 50])

    ax.spines['bottom'].set_bounds(-.5, 5.5)
    ax.set_xlim((-.7, 5.5))
    xticks = np.arange(-.5, 6, .5)
    xticklabels = ['', '0', '', '1', '', '2', '', '3', '', '4', '', '5', '']
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='both', which='major', pad=4)

    for bar_ind in range(len(hist_out[0])):
        ax.text(hist_out[1][bar_ind]+.5, hist_out[0][bar_ind]+.005, str(int(round(100.*hist_out[0][bar_ind])))+'%', horizontalalignment='center')

    #ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    file_name = 'performance_hist.svg'
    #file_name = 'performance_hist_tm2.svg'
    #file_name = 'performance_hist4D.svg'
    fig.savefig(file_name)

    fix_mpl_svg.replace(file_name)
    return fig

