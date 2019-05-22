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


# In[14]:


doros.shape


# In[9]:


doros.describe()


# In[10]:


df=pd.DataFrame(doros['doros'],columns=[''])


# In[6]:


df


# In[11]:


df.shape


# In[26]:


#https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_to_Speed-up_Machine_Learning_Algorithms.ipynb
from sklearn.preprocessing import StandardScaler #from Github PCA + Logistic regression (MNIST)
scaler = StandardScaler()
scaler.fit(doros) #fit onto dataset
doros = scaler.transform(doros) #Apply tranform onto dataset


# In[27]:


from sklearn.decomposition import PCA # import sklearn 


# In[28]:


pca = PCA(.95)


# In[29]:


pca.fit(doros)


# In[30]:


doros.shape


# In[31]:


pca.n_components


# In[16]:


#x=StandardScaler().fit_transform(df.value)


# In[10]:


#doros_pca = decomposition.PCA(n_components=0.7)# create a pca
principalComponents = doros_pca.fit_transform(x) # fit the pca to this dataset
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['1', '10000'])# save the output 


# In[ ]:





# In[ ]:




