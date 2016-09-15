import os
import numpy as np
import h5py

TEST_FRAC = .2
DATA_PATH = '../data/'

data_file_names = os.listdir(DATA_PATH)
data_file_names.sort()

vids = list(set([dfn.split('.')[0] for dfn in data_file_names]))

vid_list_file = h5py.File('../vid_list.hdf5', 'w')
vid_list_file.create_dataset('vids', data=vids)

num_vids = len(vids)

test_set_size = int(round(num_vids*TEST_FRAC))

rand_inds = np.random.permutation(num_vids)

is_test = np.zeros(num_vids, dtype='bool')
is_test[rand_inds[:test_set_size]] = True

vid_list_file.create_dataset('is_test', data=is_test)
vid_list_file.flush()
vid_list_file.close()
