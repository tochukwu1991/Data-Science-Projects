#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 'id','diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se','texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst','compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

df = pd.read_csv("breast__cancer.txt", index_col= 0)
print(df.head(10))


# In[3]:


#To check for null_values
df.info()


# In[6]:


import matplotlib.pyplot as plt
from IPython.display import display
pd.options.display.max_columns = None
import missingno as msno

print(msno.bar(df))
#axis([0, 80, 0, 120])


# In[5]:



print(df["bare nuclei"].value_counts())


# ## Exploratory Data Analysis

# In[6]:


corr = df.corr()
corr.style.background_gradient(cmap='coolwarm', axis=None)


# In[7]:


df = df[pd.to_numeric(df['bare nuclei'], errors='coerce').notnull()]
print(df["bare nuclei"].value_counts())

df['bare nuclei'] = df.loc[ :,'bare nuclei'].astype('int64')
df.dtypes


# In[8]:


#to see the summary statistics
df.describe()


# # DATA VISUALIZATION

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df_vis = df.copy()
df_vis.head()


# In[11]:


def label(element):
    if element == 2:
        return 'Bening'
    else:
        return 'Malignant'
    
class_value = df_vis['class'].map(label)
df_vis['class']= class_value

print(df_vis.head(10))

print(df_vis.shape)


# In[12]:


Bening_value = df_vis[df_vis["class"]== "Bening"].shape[0]
Malignant_value = df_vis[df_vis["class"]== "Malignant"].shape[0]
print(Bening_value )
print(Malignant_value)


objects = ('Benign', 'Malignant')
bar_position = np.arange(len(objects))
bar_height = [Bening_value,Malignant_value]

plt.bar(bar_position, bar_height, align='center', alpha=0.5, color = 'red')
plt.xticks(bar_position, objects)
plt.ylabel('Samples Number')
plt.title('Number of Benign and Malignant Samples')

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm', axis=None)
# In[13]:


sns.set()
with sns.plotting_context("notebook", font_scale=1.2):
    sns.pairplot(df_vis,vars=['clump-thickness', 'uniformity-of-cell-size', 'uniformity-of-cell-shape', 'marginal adhension', 'single epi cell size'],hue='class')

plt.show()


# In[14]:


#step 11 (data standardization)
from sklearn.preprocessing import StandardScaler
#Seperating out the features
 
df_prep = df.copy()
x = np.array(df_prep.drop(['class'],axis= 1))

# Seperating out the target
Y = np.array(df_prep['class'])

#Standardizing the features 
X = StandardScaler().fit_transform(x)
# print(X[0:5])


# In[15]:


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3' ])
print(pca.explained_variance_ratio_)
new_df = np.array(principalDf)
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_df[Y == 2][:, 0], new_df[Y == 2][:, 1],new_df[Y == 2][:, 2], color='blue', label='Bening')
ax.scatter(new_df[Y == 4][:, 0], new_df[Y == 4][:, 1], new_df[Y == 4][:, 2], color='red', label='Malignant')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('principal Component 2')
ax.set_zlabel('principal component 3')
plt.title("Three component PCA")
plt.legend();


# In[16]:


from sklearn.manifold import TSNE
import seaborn as sns
print(X)
tsne = TSNE(n_components=2, verbose=1, perplexity=38, n_iter=300)
tsne_results = tsne.fit_transform(X)
principalDf = pd.DataFrame(data = tsne_results 
             , columns = ['principal component 1', 'principal component 2'])
#print(principalDf.head(10))


new_df = np.array(principalDf)
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, projection= '3d')
ax.scatter(new_df[Y == 2][:, 0], new_df[Y == 2][:, 1], color='blue', label='Bening')
ax.scatter(new_df[Y == 4][:, 0], new_df[Y == 4][:, 1], color='red', label='Malignant')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('principal Component 2')
plt.title("two component tsne")
plt.legend();


# # Cluster Analysis

# In[17]:


#Kmeans
from sklearn.cluster import KMeans
kmns = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=10, copy_x=True, n_jobs=-1, algorithm='auto')
kY = kmns.fit_predict(X)
new_Ky =[]

def change(element):
    if element == 0:
        element = 2
    else:
        element = 4
    new_Ky.append(element)
    
for element in kY:
    change(element)
kmeans_y = np.array(new_Ky)


# In[18]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.scatter(tsne_results[:,0],tsne_results[:,1],  c=kY,cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax1.set_title('k-means clustering plot')

ax2.scatter(tsne_results[:,0],tsne_results[:,1],  c = Y, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax2.set_title('Actual clusters')
count = 0
for x in range(len(Y)):
    if Y[x] == kmeans_y [x]:
        count = count + 1
accuracy = count/(len(Y))
print(accuracy)


# In[19]:


from sklearn.cluster import SpectralClustering


# Play with gamma to optimize the clustering results
kmns = SpectralClustering(n_clusters=2,  gamma=0.5, affinity='rbf', eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)
kS = kmns.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(tsne_results[:,0],tsne_results[:,1],  c=kS, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax1.set_title('Spectral clustering plot')

ax2.scatter(tsne_results[:,0],tsne_results[:,1],  c = Y, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax2.set_title('Actual clusters')

spectral_Ky =[]
def change(element):
    if element == 0:
        element = 2
    else:
        element = 4
    spectral_Ky.append(element)
    
for element in kS:
    change(element)
spectral_y = np.array(spectral_Ky)

count = 0
for x in range(len(Y)):
    if Y[x] == spectral_y[x]:
        count = count + 1
accuracy = count/(len(Y))
print(accuracy)


# In[20]:


from sklearn.cluster import AgglomerativeClustering
aggC = AgglomerativeClustering(n_clusters=2, linkage='ward')
kA = aggC.fit_predict(X)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(tsne_results[:,0],tsne_results[:,1],  c=kA, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax1.set_title('Hierarchical clustering plot')

ax2.scatter(tsne_results[:,0],tsne_results[:,1],  c = Y, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax2.set_title('Actual clusters')

Agglo_Ky =[]
def change(element):
    if element == 0:
        element = 2
    else:
        element = 4
    Agglo_Ky.append(element)
    
for element in kA:
    change(element)
Agglo_y = np.array(Agglo_Ky)

count = 0
for x in range(len(Y)):
    if Y[x] == Agglo_y[x]:
        count = count + 1
accuracy = count/(len(Y))
print(accuracy)


# In[21]:


#clustering using gmm

from sklearn.mixture import GaussianMixture


Gaus = GaussianMixture(n_components=2)
kG = Gaus.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(tsne_results[:,0],tsne_results[:,1],  c=kG, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax1.set_title('Hierarchical clustering plot')

ax2.scatter(tsne_results[:,0],tsne_results[:,1],  c = Y, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax2.set_title('Actual clusters')

Gaus_Ky =[]

def change(element):
    if element == 0:
        element = 2
    else:
        element = 4
    Gaus_Ky.append(element)
    
for element in kG:
    change(element)
Gaus_y = np.array(Gaus_Ky)

count = 0
for x in range(len(Y)):
    if Y[x] == Gaus_y[x]:
        count = count + 1
accuracy = count/(len(Y))
print(accuracy)


# In[31]:


K_means_C = 0.96
Spectral_C = 0.93
Aglomerative_C = 0.97
GMM_C = 0.87

objects = ('K_means_C', 'Spectral_C', 'Aglomerative_C ', 'GMM_C')
bar_position = np.arange(len(objects))
bar_height = [K_means_C, Spectral_C,Aglomerative_C , GMM_C]
xlocs, xlabs = plt.xticks()

bars = plt.bar(bar_position, bar_height, align='center', alpha=0.95, color = 'pink')
plt.xticks(bar_position, objects, rotation=90)
plt.ylabel('Accuracy')
plt.title('Accuracy vs Clustering Algorithms')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval)
plt.show()


# # Exploring Agglomerative clustering

# In[34]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Breast Cancer Dendograms")
plt.ylabel('Distance')
dend = shc.dendrogram(shc.linkage(X, method='ward'))


# In[24]:


#choose 5 clusters because the they have the small dismillarity values. 

aggC = AgglomerativeClustering(n_clusters=5, linkage='ward')
kA = aggC.fit_predict(X)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


ax1.scatter(tsne_results[:,0],tsne_results[:,1],  c=kA, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax1.set_title('Hierarchical clustering plot')

ax2.scatter(tsne_results[:,0],tsne_results[:,1],  c = Y, cmap = "rainbow", edgecolor = "None", alpha=0.65)
ax2.set_title('Actual clusters')


# In[25]:


#assigning the new colum to the dataframe
adf = pd.read_csv("breast__cancer.txt")
adf = adf.drop(['ID'], axis= 1)
adf = adf[pd.to_numeric(adf['bare nuclei'], errors='coerce').notnull()]
adf['bare nuclei'] = adf.loc[ :,'bare nuclei'].astype('int64')
a_y = pd.DataFrame({'Clusters_5': kA[0:,]})
adf['Clusters_5'] = a_y
print(adf.head(5))


# In[26]:


# Manipulating the data set
grouped = adf.groupby('Clusters_5')
class_0 = grouped.get_group(0.0)
class_0_class = class_0['class'].value_counts()
print(class_0_class)

class_1 = grouped.get_group(1.0)
class_1_class = class_1['class'].value_counts()
print(class_1_class)

class_2 = grouped.get_group(2.0)
class_2_class = class_2['class'].value_counts()
print(class_2_class)

class_3 = grouped.get_group(3.0)
class_3_class = class_3['class'].value_counts()
print(class_3_class)

class_4 = grouped.get_group(4.0)
class_4_class = class_4['class'].value_counts()
print(class_4_class)
# print(grouped.size())
# y= grouped.mean()
# print(y)


# In[27]:


ax = y.plot(kind='barh', title ="V comp",figsize=(15,10),legend=True, fontsize=12)
ax.set_xlabel("Hour",fontsize=12)
ax.set_ylabel("V",fontsize=12)


# import matplotlib.pyplot as plt
# ax = df[['V1','V2']].plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)
# ax.set_xlabel("Hour", fontsize=12)
# ax.set_ylabel("V", fontsize=12)
plt.show()


# In[ ]:


values = grouped.mean()
bar_height = values.iloc[[0.0]].values
bar_heights_0 = []
for x in bar_height:
    for e in x:
        bar_heights_0.append(round(e,2))

objects = ('clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class')
bar_position = np.arange(len(objects))
xlocs, xlabs = plt.xticks()

# reference x so you don't need to change the range each time x changes
xlocs=[i for i in bar_position]
xlabs=['clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class']
#bar_height = [Bening_value,Malignant_value]

bars =plt.bar(bar_position, bar_heights_0, align='center', alpha=0.7, color = 'red')
plt.xticks(bar_position, objects, rotation=90)
# plt.xticks(xlocs, xlabs)
plt.ylabel('Feature Values')
plt.title('Values for each feature in class 0')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval)


# In[ ]:


values = grouped.mean()
bar_height = values.iloc[[1.0]].values
bar_heights_1 = []
for x in bar_height:
    for e in x:
        bar_heights_1.append(round(e,2))

objects = ('clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class')
bar_position = np.arange(len(objects))
xlocs, xlabs = plt.xticks()

# reference x so you don't need to change the range each time x changes
xlocs=[i for i in bar_position]
xlabs=['clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class']
#bar_height = [Bening_value,Malignant_value]

bars =plt.bar(bar_position, bar_heights_1, align='center', alpha=0.7, color = 'purple')
plt.xticks(bar_position, objects, rotation=90)
# plt.xticks(xlocs, xlabs)
plt.ylabel('Feature Values')
plt.title('Values for each feature in class 1')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Show metrics 
def show_metrics():
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision =     {:.3f}'.format(tp/(tp+fp)))
    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))
    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/
                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))


# In[ ]:


values = grouped.mean()
bar_height = values.iloc[[2.0]].values
bar_heights_2 = []
for x in bar_height:
    for e in x:
        bar_heights_2.append(round(e,2))

objects = ('clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class')
bar_position = np.arange(len(objects))
xlocs, xlabs = plt.xticks()

# reference x so you don't need to change the range each time x changes
xlocs=[i for i in bar_position]
xlabs=['clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class']
#bar_height = [Bening_value,Malignant_value]

bars =plt.bar(bar_position, bar_heights_2, align='center', alpha=0.7, color = 'blue')
plt.xticks(bar_position, objects, rotation=90)
# plt.xticks(xlocs, xlabs)
plt.ylabel('Feature Values')
plt.title('Values for each feature in class 2')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval)


# In[ ]:


values = grouped.mean()
bar_height = values.iloc[[3.0]].values
bar_heights_3 = []
for x in bar_height:
    for e in x:
        bar_heights_3.append(round(e,2))

objects = ('clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class')
bar_position = np.arange(len(objects))
xlocs, xlabs = plt.xticks()

# reference x so you don't need to change the range each time x changes
xlocs=[i for i in bar_position]
xlabs=['clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class']
#bar_height = [Bening_value,Malignant_value]

bars =plt.bar(bar_position, bar_heights_3, align='center', alpha=0.7, color = 'orange')
plt.xticks(bar_position, objects, rotation=90)
# plt.xticks(xlocs, xlabs)
plt.ylabel('Feature Values')
plt.title('Values for each feature in class 1')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval)


# In[ ]:


values = grouped.mean()
bar_height = values.iloc[[4.0]].values
bar_heights_4 = []
for x in bar_height:
    for e in x:
        bar_heights_4.append(round(e,2))

objects = ('clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class')
bar_position = np.arange(len(objects))
xlocs, xlabs = plt.xticks()

# reference x so you don't need to change the range each time x changes
xlocs=[i for i in bar_position]
xlabs=['clump-thickness', 'uniformity-cell-size','uniformity-cell-shape','marginal adhension',
           'single epi cell size','bare nuclei', 'bland chromatin','Normal Nucleoli', 'mitosis','class']
#bar_height = [Bening_value,Malignant_value]

bars =plt.bar(bar_position, bar_heights_4, align='center', alpha=0.7, color = 'green')
plt.xticks(bar_position, objects, rotation=90)
# plt.xticks(xlocs, xlabs)
plt.ylabel('Feature Values')
plt.title('Values for each feature in class 1')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval)


# ### conclusion 
# 1. class 0 - represents people who are very benign.
# 2. class 1- represents people who are in stage 1 or 2 of cancer
# 3. class 2- represents peoplw whoa are benign but still they are very prone to being malignant
# 4. class 3- represents people who are like in the stage 3 of the cancer
# 5. class 4- represents people who are in stage 4 of cancer.

# In[ ]:




