#%%
# find out how many sets of data are in the file
import h5py
f = h5py.File('/home/goodvibrationskde/Documents/hdf5_sample_data.h5', 'r')
f.keys()
print(f.keys())
f.close()


# %%
