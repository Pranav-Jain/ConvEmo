from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
import csv
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import completeness_score
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d


max_epochs = 10
vec_size = 300
alpha = 0.0002

data = []
labels = []

d = {'happy':0, 'sad':1, 'angry':3, 'others':4}
colors = ['blue','green','red','yellow']

line_count = 0

################################ Read data file ###################################
with open('train.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
    	if(line_count!=0):
    		data.append(row[1])
    		data.append(row[2])
    		data.append(row[3])
    		labels.append(d[row[4]])
    	line_count+=1
###################################################################################

############################### Create tags #######################################
tag = []
for i in range(line_count):
	tag.append(i)
	tag.append(i)
	tag.append(i)
###################################################################################

############################## Vectorise using Doc2Vec ############################
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(tag[i])]) for i, _d in enumerate(data)]
model = Doc2Vec(vector_size=vec_size, window=5, min_count=0, workers=4)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
	print('Epoch {0}'.format(epoch))
	model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
	model.alpha -= alpha # decrease the learning rate
	model.min_alpha = model.alpha # fix the learning rate, no decay
model.save("d2v.model")
print("Model Saved")
####################################################################################