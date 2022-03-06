#%%
# find out how many sets of data are in the file
import h5py
f = h5py.File('/home/goodvibrationskde/Documents/Large_files/hdf5_sample_data.h5', 'r')
f.keys()
f.close()


# %%
