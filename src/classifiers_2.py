from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


def svm(X_train, X_test, y_train, y_test):
    clf = SVC(C=32768, kernel='rbf', gamma=3.05e-5)
    clf.fit(X_train, y_train)

    print("SVM Trained")

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))

    print("Test Accuracy")
    print(accuracy_score(y_test, pred))

    pred = clf.predict(X_train)
    print("Train Accuracy")
    print(accuracy_score(y_train, pred))

    pickle.dump(clf, open('svm2.sav', 'wb+'))
    print("Model Saved")

def nb(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    print("Naive Bayes Model Trained")

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))

    print("Test Accuracy")
    print(accuracy_score(y_test, pred))

    pred = clf.predict(X_train)
    print("Train Accuracy")
    print(accuracy_score(y_train, pred))

    pickle.dump(clf, open('nb2.sav', 'wb+'))
    print("Model Saved")


def nn(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(solver='adam', alpha=1,hidden_layer_sizes=(100, 25))
    clf.fit(X_train, y_train)

    print("Neural Net Trained")

    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))

    print("Test Accuracy")
    print(accuracy_score(y_test, pred))

    pred = clf.predict(X_train)
    print("Train Accuracy")
    print(accuracy_score(y_train, pred))

    pickle.dump(clf, open('nn2.sav', 'wb+'))
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
                labels.append(d[row[4]])
            line_count+=1

    train_vector = np.load('data_vectors_2.npy')

    X_train, X_test, y_train, y_test = train_test_split(train_vector, labels, test_size=0.2)
    print("Data Split Done....\n")

    # np.save('X_train', X_train)
    # np.save('X_test', X_test)
    # np.save('y_train', y_train)
    # np.save('y_test', y_test)


    print("\nSVM Model")
    svm(X_train, X_test, y_train, y_test)

    print("\nNaive Bayes Model")
    nb(X_train, X_test, y_train, y_test)

    # print("\nLogistic Regression Model")
    # lr(X_train, X_test, y_train, y_test)

    print("\nNeural Net")
    nn(X_train, X_test, y_train, y_test)


