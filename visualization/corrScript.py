import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

db = pd.read_csv('income_stat_all_us_list.csv')
db = db.drop(['gvkey','datadate','indfmt','consol','popsrc','datafmt'],1)
db = db.drop(['curcd','batr','bct','bctr','citotal','cik','cga','cgti'],1)
db = db.drop(['ibki','initb','ipti','isgt','ivi','li','nfsr',
              'niit','niint','nim','nit','nits','opili','opiti','patr'],1)
db = db.drop(['ptbl','rcp','spce','stkco','tie','tii','txds','xlr','xnitb',
              'xoprar','xrd','xt','xuwti','costat'],1)
db = db.drop(['tic'],1)
# Removes variables non relevant to data or with over 50% of values missing

comName = db.conm.unique()
varDict = db.columns.values
varDict = np.delete(varDict, [0,1])
comData = {}
corrDict = {}
count = 0
#corrArray = np.zeros(shape=(22,22,1),dtype=np.float32)
# Gets company names, column names, and initializes an empty dictionary

def cleanRow(x):
    headName = x.columns.values
    x = x.dropna(subset=headName)
    return x
# Function to clean rows (Removes NaNs)

def getTwoColumns(database,j,k):
    xByTwo = database[[database.columns[j],database.columns[k]]]
    return xByTwo
# Gets columns j and k in a database

for i in comName[0:10]:
    comData[i] = db.loc[db['conm'] == i]
    comData[i] = comData[i].drop(['conm','fyear'],1)
    columnName = comData[i].columns.values
    corrDB = pd.DataFrame(columns=varDict,index=varDict)
    for j in xrange(len(columnName)-1):
        for k in xrange((len(columnName)-j)-1):
            test = getTwoColumns(comData[i],j,k+j+1)
            test = cleanRow(test)
            correlation = test.corr(method='pearson')
            corr = correlation.iat[1,0]            
            corrDB.at[correlation.columns[0],correlation.columns[1]] = corr            
    
    corrMatrix = corrDB.as_matrix()
    print(count)
    corrMatrix.reshape(22,22,1)
    if count == 0:
        corrArray = corrMatrix
    else:
        corrArray = np.dstack((corrArray, corrMatrix))
    #corrArray = corrArray[~np.isnan(corrArray)]
    count += 1

corrArray = corrArray.astype(np.float32)

corrArray = np.nanmean(corrArray, axis=2)

mask = np.zeros_like(corrArray, dtype=np.bool)
mask[np.tril_indices_from(mask)] = True
corrArray[np.tril_indices_from(mask)] = 1

sns.set(style="white")
cmap = sns.diverging_palette(220, 10, as_cmap=True)
#plt.subplots(figsize=(20,15))
ax = plt.axes()
# Draw the heatmap with the mask and correct aspect ratio
var1 = sns.heatmap(corrArray, ax=ax, mask=mask, cmap=cmap,annot=True, annot_kws={"size": 3.5}, 
                   fmt=".2g",linewidths=.5,xticklabels=varDict,yticklabels=varDict)
ax.set_title('Income Statistics Correlation Matrix')
fig1 = var1.get_figure()
fig1.savefig("corr.png", dpi=800)
