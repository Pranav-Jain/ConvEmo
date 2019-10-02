
# coding: utf-8

# <h1><center>ConvEmo</h1>

# In[1]:

import numpy as np
import pandas as pd
import csv
import codecs


# ### Data Processing

# In[2]:

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


# In[4]:

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

sid = SentimentIntensityAnalyzer()

turn1_sentiment = []
turn2_sentiment = []
turn3_sentiment = []


# ### Getting the Sentiments

# #### Sentiment - Turn 1

# In[5]:

for t in turn1:
    s = sid.polarity_scores(t)
    compound = s['compound']
    turn1_sentiment.append(compound)


# #### Sentiment - Turn 2

# In[6]:

for t in turn2:
    s = sid.polarity_scores(t)
    compound = s['compound']
    turn2_sentiment.append(compound)


# #### Sentiment - Turn 3

# In[7]:

for t in turn3:
    s = sid.polarity_scores(t)
    compound = s['compound']
    turn3_sentiment.append(compound)


# #### Preparing numPy array

# In[8]:

X = np.array(list(zip(turn1_sentiment, turn2_sentiment)))
y = np.array(labels)


# ### Visualizing the Sentiments

# In[62]:

# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (12,12)


# In[63]:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(turn1_sentiment),np.array(turn2_sentiment),np.array(turn3_sentiment), c=y, alpha=0.5)
plt.show()


# ### Visualizing the Sentiments using Plotly

# In[70]:

import plotly
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


# In[92]:

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
for i in range(len(turn1_sentiment)):
    if(labels[i]==0):
        t1_0.append(turn1_sentiment[i])
        t2_0.append(turn2_sentiment[i])
        t3_0.append(turn3_sentiment[i])
    if(labels[i]==1):
        t1_1.append(turn1_sentiment[i])
        t2_1.append(turn2_sentiment[i])
        t3_1.append(turn3_sentiment[i])
    if(labels[i]==3):
        t1_3.append(turn1_sentiment[i])
        t2_3.append(turn2_sentiment[i])
        t3_3.append(turn3_sentiment[i])
    if(labels[i]==4):
        t1_4.append(turn1_sentiment[i])
        t2_4.append(turn2_sentiment[i])
        t3_4.append(turn3_sentiment[i])


# *TSNE* has been plotted in 3-D Scatter Plot using Plotly. [https://plot.ly/python/3d-scatter-plots/]

# In[104]:

data = [
    go.Surface(z=np.array(X)),
]

plotly.offline.plot(data,filename='multiple-surfaces')


# In[100]:

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
data = [trace0,trace1,trace3]
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


# ### Using SVM for Classification (2 feature : turn1 and turn2)

# In[9]:

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[10]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[11]:

model = svm.SVC(gamma='auto')
model.fit(X_train,y_train)


# In[13]:

print("Train Score : ",model.score(X_train,y_train))
print("Test Score : ",model.score(X_test,y_test))

y_pred = model.predict(X_test)
print("Test Accuracy Score : ",accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


# ### Using SVM for Classification (3 feature : turn1 and turn2 and turn3)

# In[14]:

X = np.array(list(zip(turn1_sentiment, turn2_sentiment,turn3_sentiment)))
y = np.array(labels)


# In[15]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[16]:

model = svm.SVC(gamma='auto')
model.fit(X_train,y_train)


# In[17]:

print("Train Score : ",model.score(X_train,y_train))
print("Test Score : ",model.score(X_test,y_test))

y_pred = model.predict(X_test)
print("Test Accuracy Score : ",accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


# ### Using SVM for Classification 5 with fold CV (3 feature : turn1 and turn2 and turn3)

# In[18]:

from sklearn.model_selection import cross_val_score


# In[19]:

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X , y , cv=5)


# In[20]:

i=1
for s in scores:
    print("CV ",str(i),"\t:  ",s)
print()
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ### Using Logistic Regression for Classification 5 with fold CV (3 feature : turn1 and turn2 and turn3)

# In[21]:

from sklearn.linear_model import LogisticRegression


# In[22]:

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')


# In[23]:

clf.fit(X_train,y_train)


# In[24]:

print("Train Score : ",clf.score(X_train,y_train))
print("Test Score : ",clf.score(X_test,y_test))


# In[25]:

y_pred = clf.predict(X_test)
print("Test Accuracy Score : ",accuracy_score(y_test, y_pred))


# In[26]:

print(classification_report(y_test, y_pred))


# In[27]:

w = clf.coef_
w

