import numpy as np
import pandas as pd

whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
flavours = whisky.iloc[:, 2:14]


#correlating data

corr_flavours = pd.DataFrame.corr(flavours)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavours)
plt.colorbar()
plt.savefig("corr_flavours.pdf")

corr_whisky = pd.DataFrame.corr(flavours.transpose())
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.savefig("corr_whisky.pdf")

#clustering data - spectral coclustering method

from sklearn.cluster.bicluster import SpectralCoclustering as scc

model = scc(n_clusters = 6, random_state = 0) #there are 6 regions
model.fit(corr_whisky)

np.sum(model.rows_, axis =1)


#comparing correlation matricies
whisky['Group'] = pd.Series(model.row_labels_, index =whisky.index)
whisky = whisky.iloc[np.argsort(model.row_labels_)]  #sorts the df by passing it a list of the indexes of the sorted columns
whisky = whisky.reset_index(drop=True) # resets the index labelling after the sorting
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)  #want a np array instead of DF for plotting


plt.figure(figsize=(10,10))
plt.pcolor(correlations)
plt.colorbar()
plt.savefig("correlations.pdf")