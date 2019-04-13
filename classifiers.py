from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def svm(X_train, X_test, y_train, y_test):
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)

    print("SVM Trained")

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    pickle.dump(clf, open('svm.sav', 'wb+'))
    print("Model Saved")

def knn(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, y_train)

    print("KNN Model Trained")

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    pickle.dump(clf, open('knn.sav', 'wb+'))
    print("Model Saved")

def lr(X_train, X_test, y_train, y_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    print("Logistic Regression Model Trained")

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    pickle.dump(clf, open('lr.sav', 'wb+'))
    print("Model Saved")

def nn(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, ))
    clf.fit(X_train, y_train)

    print("Neural Net Trained")

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    pickle.dump(clf, open('nn.sav', 'wb+'))
    print("Model Saved")


if __name__ == '__main__':


    d = {'happy':0, 'sad':1, 'angry':2, 'others':3}
    colors = ['blue','green','red','yellow']

    data = []
    labels= []

    with open('train.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        for row in csv_reader:
            if(line_count!=0):
                string = row[1]
                for i in range(2):
                    string = string + " " + row[i+2]
                data.append(string)
                labels.append(d[row[4]])
            line_count+=1


    model_test= Doc2Vec.load("d2v.model")

    train_vector = []

    for i in data:
        t = word_tokenize(i.lower())
        v1 = model_test.infer_vector(t)
        # v1 = preprocessing.scale(v1)
        train_vector.append(v1)

    X_train, X_test, y_train, y_test = train_test_split(train_vector, labels, test_size=0.2)
    print("Data Split Done....\n")

    print("\nSVM Model")
    svm(X_train, X_test, y_train, y_test)

    print("\nKNN Model")
    knn(X_train, X_test, y_train, y_test)

    print("\nLogistic Regression Model")
    lr(X_train, X_test, y_train, y_test)

    print("\nNeural Net")
    nn(X_train, X_test, y_train, y_test)


