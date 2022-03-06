#%%
import pandas as pd
nRows= 0
for dfChunk in pd.read_csv('generic-food.csv',chunksize=50):
    nRows=nRows + len(dfChunk)
    print ("processed {0}".format(nRows))
    print (dfChunk)


# %%
import pandas as pd

df = pd.read_csv('/home/goodvibrationskde/Documents/5m Sales Records.csv')


#%%
#Returns first 5 rows of dataset in presentation mode 
df.head()

#%%
#Information of the dataframe in the csv
df.info()

#%%
#Search for spesific column in the csv file and return 1-D array
df['Country']

#Alternative way to return 1-D array
df.Country

#%%
#Read a csv file by columns
df_by_order_ID= pd.read_csv('/home/goodvibrationskde/Documents/5m Sales Records.csv', usecols=['Total Profit'])

print (df_by_order_ID)

df_by_order_ID.info()


#%%
#Subset of dataset and return all rows from specified keys
subset = df.loc [:, ['Region', 'Country', 'Total Profit']]
subset.head()

#%%
#Subset of dataset and return firts 5 rows
subset = df [['Region', 'Country', 'Total Profit']]
subset.head()

#%%
#Returns the total shape of the dataframe
df.shape


# %%
