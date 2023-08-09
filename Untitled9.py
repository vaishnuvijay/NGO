#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/vaish/OneDrive/Desktop/NGO.csv")
df


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


fig= plt.figure(figsize=(20,5))
fig.set_facecolor("#F3F3F3")
a=sns.boxplot(data=df)
plt.setp(a.get_xticklabels(), rotation=45)
plt.title('Box-Plot', fontsize=14)
plt.ylabel('Values', fontsize=18);plt.xlabel('Parameters',fontsize=14)
plt.yticks(fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.show()


# In[ ]:





# In[ ]:





# In[12]:


g=df.corr()
b=sns.heatmap(g,cmap="YlGnBu",annot = True)
plt.setp(b.get_xticklabels(), rotation=45)
plt.yticks(fontsize=14, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.show()


# In[17]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df1=df.drop('country',1)
scaler=StandardScaler()
country_scaled = scaler.fit_transform(df1)


# In[18]:


pca = PCA(svd_solver='randomized', random_state=42)
pca.fit(country_scaled)


# In[19]:


pca.components_


# In[20]:


pca.explained_variance_ratio_


# In[22]:


import numpy as np
fig = plt.figure(figsize = (12,5))
fig.set_facecolor("#F3F3F3")
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Scree plot:The Cumulative Variance Against The Number of Components', fontsize=14)
plt.xlabel('The Number of Components', fontsize=12, fontweight='bold')
plt.ylabel('The Cumulative Explained Variance', fontsize=12,fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')


# In[23]:


colnames = list(df1.columns)
prinComp_df = pd.DataFrame({ 'Feature':colnames,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2],'PC4':pca.components_[3],'PC5':pca.components_[4]})
prinComp_df


# In[24]:


fig = plt.figure(figsize = (10,18))
fig.set_facecolor("#F3F3F3")
plt.scatter(prinComp_df.PC1, prinComp_df.PC2)
plt.xlabel('Principal Component 1',fontsize=12, fontweight='bold')
plt.ylabel('Principal Component 2',fontsize=12, fontweight='bold')
for i, txt in enumerate(prinComp_df.Feature):
    plt.annotate(txt, (prinComp_df.PC1[i],prinComp_df.PC2[i]),fontsize=12,  horizontalalignment='right')
plt.yticks(fontsize=12, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.show()


# In[26]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=42)
    kmeans.fit(df1)
    wcss.append(kmeans.inertia_)
    wcss


# In[27]:


wcss


# In[28]:


import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss,marker="*")


# In[29]:


kmean=KMeans(n_clusters=3,init="k-means++",random_state=42)
pre=kmean.fit_predict(df1)
pre


# In[30]:


df1["cluster"]=pre


# In[31]:


m=kmean.cluster_centers_
m


# In[32]:


m[:,0]


# In[40]:


import seaborn as sns
plt.figure(figsize=(10,10))
sns.scatterplot(x=df["gdpp"],y=df["imports"],hue=df1["cluster"])

plt.xticks(rotation="vertical")


# In[ ]:




