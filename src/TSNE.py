
# coding: utf-8

# # TSNE

# ##### TSNE has been used to reduce the dimension of doc2vec embeddings from 300 to 3

# *TSNE* has been plotted in 3-D Scatter Plot using Plotly. [https://plot.ly/python/3d-scatter-plots/]

# In[14]:

import numpy as np
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import csv
import codecs


# In[15]:

turn1 = []
turn2 = []
turn3 = []
labels = []
line_count = 0

d = {'happy':0, 'sad':1, 'angry':3, 'others':4}

with codecs.open('train.txt',encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        if(line_count!=0):
            turn1.append(row[1])
            turn2.append(row[2])
            turn3.append(row[3])
            labels.append(d[row[4]])
        line_count+=1


# In[16]:

vect = np.load('data_vectors.npy')


# In[17]:

np.shape(vect)


# In[18]:

x= vect.shape[0]
x


# In[4]:

v = TSNE(n_components=3).fit_transform(vect)


# In[19]:

np.shape(v)


# In[20]:

np.save('tsne', v)


# In[25]:

t1_0 = []
t1_1 = []
t1_3 = []
t1_4 = []
t2_0 = []
t2_1 = []
t2_3 = []
t2_4 = []
t3_0 = []
t3_1 = []
t3_3 = []
t3_4 = []

l = v.shape[0]
for i in range(0,l):
    if(labels[i]==0):
        t1_0.append(v[i][0])
        t2_0.append(v[i][1])
        t3_0.append(v[i][2])
    if(labels[i]==1):
        t1_1.append(v[i][0])
        t2_1.append(v[i][1])
        t3_1.append(v[i][2])
    if(labels[i]==3):
        t1_3.append(v[i][0])
        t2_3.append(v[i][1])
        t3_3.append(v[i][2])
    if(labels[i]==4):
        t1_4.append(v[i][0])
        t2_4.append(v[i][1])
        t3_4.append(v[i][2])


# In[26]:

print(v[0][1])


# In[27]:

import plotly
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[29]:

x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
trace0 = go.Scatter3d(
    x=np.array(t1_0),
    y=np.array(t2_0),
    z=np.array(t3_0),
    #colors = np.array(labels),
    mode='markers',
    marker=dict(
        size=12,
        #color='rgb(255, 0, 255)',
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.1
        ),
        opacity=0.7
    )
)
trace1 = go.Scatter3d(
    x=np.array(t1_1),
    y=np.array(t2_1),
    z=np.array(t3_1),
    mode='markers',
    marker=dict(
        size=12,
        color='rgb(127, 127, 127)',
        symbol='circle',
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.1
        ),
        opacity=0.9
    )
)
trace3 = go.Scatter3d(
    x=np.array(t1_3),
    y=np.array(t2_3),
    z=np.array(t3_3),
    mode='markers',
    marker=dict(
        size=12,
        color='rgb(255, 255, 0)',
        symbol='circle',
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.1
        ),
        opacity=0.9
    )
)
trace4 = go.Scatter3d(
    x=np.array(t1_4),
    y=np.array(t2_4),
    z=np.array(t3_4),
    mode='markers',
    marker=dict(
        size=12,
        color='rgb(255, 0, 255)',
        symbol='circle',
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.1
        ),
        opacity=0.9
    )
)
data = [trace0,trace1,trace3,trace4]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='simple-3d-scatter')

