#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd      # package for handling data formats
import numpy as np       #package for scientific functions such as linear regression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage  
import matplotlib.pyplot as plt

doros = pd.read_csv("Datasets/dorothea_smaller.csv")

import random 
random.seed(42)


# In[ ]:


doros.columns()


# In[ ]:


doros.index()


# In[5]:


df=pd.DataFrame


# In[6]:


df


# In[9]:


x=StandardScaler().fit_transform(df.value)


# In[10]:


doros_pca = decomposition.PCA(n_components=2)# create a pca
principalComponents = doros_pca.fit_transform(x) # fit the pca to this dataset
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2'])# save the output 


# In[ ]:




