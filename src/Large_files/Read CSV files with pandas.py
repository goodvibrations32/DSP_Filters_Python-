"""
#%%
import pandas as pd
nRows= 0
for dfChunk in pd.read_csv('generic-food.csv',chunksize=50):
    nRows=nRows + len(dfChunk)
    print ("processed {0}".format(nRows))
    print (dfChunk)
"""

# %%
#Read a csv file by columns

import pandas as pd
df= pd.read_csv('5m Sales Records.csv', usecols=['Order ID'])
print (df)

df.info()

# %%
