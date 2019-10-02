# Reference from Online Tutorial on using doc2vec, GitHub repo and Medium 

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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

max_epochs = 10
vec_size = 100
alpha = 0.0002

data1 = []
data2 = []
data3 = []
labels = []

d = {'happy':0, 'sad':1, 'angry':2, 'others':3}
colors = ['blue','green','red','yellow']

with open('train.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        if(line_count!=0):
            data1.append(row[1])
            data2.append(row[2])
            data3.append(row[3])
            labels.append(d[row[4]])
        line_count+=1

# Reference from Online Tutorial on using doc2vec, GitHub repo and Medium 

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data1)]
model = Doc2Vec(vector_size=vec_size, window=5, min_count=0, workers=4)
model.build_vocab(tagged_data)
# fname = get_tmpfile("my_doc2vec_model")
# model.save(fname)
for epoch in range(max_epochs):
  print('Epoch '+str(epoch))
  model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
  # decrease the learning rate
  model.alpha -= alpha
  # fix the learning rate, no decay
  model.min_alpha = model.alpha
model.save("d2v_1.model")
print("Model Saved")

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data2)]
model = Doc2Vec(vector_size=vec_size, window=5, min_count=0, workers=4)
model.build_vocab(tagged_data)
# fname = get_tmpfile("my_doc2vec_model")
# model.save(fname)
for epoch in range(max_epochs):
    print('Epoch'+str(epoch))
    model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
    # decrease the learning rate
    model.alpha -= alpha
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
model.save("d2v_2.model")
print("Model Saved")

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data3)]
model = Doc2Vec(vector_size=vec_size, window=5, min_count=0, workers=4)
model.build_vocab(tagged_data)
# fname = get_tmpfile("my_doc2vec_model")
# model.save(fname)
for epoch in range(max_epochs):
    print('Epoch'+str(epoch))
    model.train(tagged_data,total_examples=model.corpus_count,epochs=model.iter)
    # decrease the learning rate
    model.alpha -= alpha
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
model.save("d2v_3.model")
print("Model Saved")

# model_test1 = Doc2Vec.load("d2v_1.model")
# model_test2 = Doc2Vec.load("d2v_2.model")
# model_test3 = Doc2Vec.load("d2v_3.model")

# test_data = []
# turn3 = []

# for i in range(len(labels)):
#     d = []
#     test1 = word_tokenize(data1[i].lower())
#     test2 = word_tokenize(data2[i].lower())
#     test3 = word_tokenize(data3[i].lower())
#     x1 = model_test1.infer_vector(test1)
#     x2 = model_test2.infer_vector(test2)
#     x3 = model_test3.infer_vector(test3)
#     test_data.append(np.concatenate((x1,x2,x3)))
#     turn3.append(x3)

# test_data = np.array(test_data)
# turn3 = np.array(turn3)
# print(test_data.shape)
# print(turn3.shape)

# np.save('data_vectors_2', test_data)
# np.save('turn3', turn3)