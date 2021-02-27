import h5py 
import matplotlib.pyplot as plt

path = "/mnt/l/USC/PRL/karel_dataset_stmt_cond_act_20d_trace_1.0_prob_2_trials_200_35k_7.5k_7.5k/data.hdf5"
file = h5py.File(path, 'r')
fig = plt.figure()
ax = plt.subplot(111)

program_keys = file.keys()

lengths = []
for program in program_keys:
    if program != 'data_info':
        lengths.extend(file[program]['a_h_len'][()])

ax.hist(lengths)