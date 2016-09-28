from __future__ import print_function
import numpy as np
import pandas as pd
import h5py, os
import fix_mpl_svg
import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 24

df = pd.read_hdf('../vid_published_datetimes.hdf5')

df.groupby('category_id').vid.nunique()

category_id_2_name = {1 : 'Film & Animation',
 2 : 'Autos & Vehicles',
 10 : 'Music',
 15 : 'Pets & Animals',
 17 : 'Sports',
 18 : 'Short Movies',
 19 : 'Travel & Events',
 20 : 'Gaming',
 21 : 'Videoblogging',
 22 : 'People & Blogs',
 23 : 'Comedy',
 24 : 'Entertainment',
 25 : 'News & Politics',
 26 : 'Howto & Style',
 27 : 'Education',
 28 : 'Science & Technology',
 29 : 'Nonprofits & Activism',
 30 : 'Movies',
 31 : 'Anime/Animation',
 32 : 'Action/Adventure',
 33 : 'Classics',
 34 : 'Comedy',
 35 : 'Documentary',
 36 : 'Drama',
 37 : 'Family',
 38 : 'Foreign',
 39 : 'Horror',
 40 : 'Sci-Fi/Fantasy',
 41 : 'Thriller',
 42 : 'Shorts',
 43 : 'Shows',
 44 : 'Trailers'}

fig = plt.figure()
ax = fig.add_subplot(111)

si = np.argsort(df.groupby('category_id').vid.nunique().values)[::-1]
ticklabels = [category_id_2_name[ind] for ind in df.groupby('category_id').vid.nunique().index[si]]
ax.barh(np.arange(len(si)), df.groupby('category_id').vid.nunique().values[si], tick_label = ticklabels)

ax.set_ylabel('Category', fontname='FreeSans')
ax.set_xlabel('# videos', fontname='FreeSans')

labels = ax.get_yticklabels()
#plt.setp(labels, rotation=-50, ha='left')
plt.setp(labels, va='bottom')

ax.tick_params(axis='both', which='major', pad=4)
ax.tick_params(axis='y', length=0)

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_ylim(-.5, 15)
ax.set_xlim(0, 1000)

fig.savefig('categories.svg')
fix_mpl_svg.replace('categories.svg')

