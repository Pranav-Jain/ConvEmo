from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score
from multiprocessing import Process
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_input(file_path):
	f = open(file_path,"r")
	s = f.read()
	s = s.strip().split("\n")
	turn1 = []
	turn2 = []
	turn3 = []
	Y = []
	dic = {"others":3,"angry":2,"sad":1,"happy":0}
	for l in s[0:]:
		l = l.strip().split("\t")
		turn1.append(l[1])
		turn2.append(l[2])
		turn3.append(l[3])
		Y.append(dic[l[4]])

	return turn1,turn2,turn3,Y

def make_vector(corpus,stop_words):
	vectorizer = CountVectorizer(stop_words = stop_words)
	X = vectorizer.fit_transform(corpus)
	return X.toarray(),vectorizer.get_feature_names()

def SVC_parallelize(X,Y,name,count,X_test,Y_test):
	clf = SVC(kernel = "rbf", decision_function_shape = 'ovr')
	model = clf.fit(X,Y)
	print(accuracy_score(model.predict(X),Y),"Training Error")
	print(accuracy_score(model.predict(X_test),Y_test),"Test Error")
	f = open(name,"wb+")
	pickle.dump(model,f)
	f.close()
	analyse(name,X,count)

def train_SVC(X_train,Y_train,X_test,Y_test):
	process = []
	for j in range(4):
		Y_temp = np.zeros(len(Y_train))
		for i in range(len(Y_train)):
			if Y_train[i] == j:
				Y_temp[i] = 1
			else:
				Y_temp[i] = 0
		Y_temp_test = np.zeros(len(Y_test))
		for i in range(len(Y_test)):
			if Y_test[i] == j:
				Y_temp_test[i] = 1
			else:
				Y_temp_test[i] = 0
		p = Process(target=SVC_parallelize, args=(X_train,Y_temp,"model"+str(j)+".bin",str(j),X_test,Y_temp_test))
		process.append(p)
		p.start()
	for l in process:
		l.join()

def train_RF(X,Y,feature):
	clf = RandomForestClassifier()
	model = clf.fit(X,Y)
	importance = clf.feature_importances_.tolist()
	temp_feature = []
	for l in feature:
		temp_feature.extend(l)
	# 	print(feature)
	triggered = find_elements(importance,20,temp_feature)
	print(triggered)



def analyse(file_path,X,count):
	model = pickle.load(open(file_path,"rb"))
	support_vectors = model.support_vectors_
	# support_ = model.support_
	dual_coef = model.dual_coef_
	# print(len(dual_coef),"dual_coef size")
	weight = np.zeros(len(support_vectors[0]))
	for l in range(len(support_vectors)):
		weight += dual_coef[0][l] * support_vectors[l]
	f = open("weight"+count+".bin","wb+")
	pickle.dump(weight,f)
	f.close()


def find_elements(weight,number,feature):
	triggered_words = []
	feature_temp = np.copy(feature)
	# print(feature_temp)
	# quit(0)
	# print(len(feature),len(weight))
	while(len(triggered_words)<number):
		max_element = np.max(weight)
		# print(max_element)
		itemindex = np.where(weight == max_element)
		# print(itemindex)
		for l2 in itemindex[0]:
			feat = feature_temp[l2]
			triggered_words.append(feat)
			# 
			#print(feature_temp[l2])
		weight = np.delete(weight,itemindex[0])
		feature_temp = np.delete(feature_temp,itemindex[0])
		# print(feature_temp)
		# print(len(weight),"weight size")
		# print(len(triggered_words))
	return triggered_words


if __name__ == '__main__':
	# corpus1 = ["I'm Aman Roy and she's not my sister","I have got my guts","i am your pain"]
	corpus_turn1,corpus_turn2,corpus_turn3,Y = load_input("./new_data.txt")
	# print(Y)
	stop_words = list(set(stopwords.words("english")))
	vectorizer = CountVectorizer(stop_words = stop_words)
	size = []
	X_turn1,feature1 = make_vector(corpus_turn1,stop_words)
	size.append(len(X_turn1[0]))
	X_turn2,feature2 = make_vector(corpus_turn2,stop_words)
	size.append(len(X_turn2[0]))
	X_turn3,feature3 = make_vector(corpus_turn3,stop_words)
	size.append(len(X_turn3[0]))
	X = np.column_stack((X_turn1,X_turn2,X_turn3))
	mean = np.mean(X,axis = 0)
	std = np.std(X,axis = 0)
	for i in range(len(X[0])):
		X[:,i] = (X[:,i]-mean[i])/std[i]
	# print(len(X))
	# print(len(X[0]))
	print("Number of Inputs",len(X))
	print("Number of Features",len(X[0]))
	print("Number of Features in each Train",size)
	print("Start Training")
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
	train_SVC(X_train,Y_train,X_test,Y_test)
	print("Training Done")
	feature = []
	feature.append(feature1)
	feature.append(feature2)
	feature.append(feature3)
	# train_RF(X,Y,feature)
	# quit(0)
	print(size[0],size[1],size[2],"sizes")
	for l in range(4):
		f = open("weight"+str(l)+".bin","rb")
		weight = pickle.load(f)
		# print(len(weight),"weight size")
		weights = []
		weights.append(weight[0:size[0]])
		weights.append(weight[size[0]:size[0]+size[1]])
		weights.append(weight[size[0]+size[1]:])
		for l in range(len(weights)):
			# print(weights[l],len(weights[l]))
			# quit(0)
			triggered_words = find_elements(weights[l],10,feature[l])
			print("turn "+str(l+1)+" triggered_words",triggered_words)
		f.close()
		
	
	# print(len(vectorizer.get_feature_names()))
