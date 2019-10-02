from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import csv
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB

def fit_report_save(X_train, X_test, y_train, y_test, clf, clf_name, param_grid):

    num_folds = 5

    GS = GridSearchCV(estimator=clf, param_grid=param_grid, cv=num_folds, refit=True, n_jobs=-1, verbose=2)
    GS.fit(X_train, y_train)

    clf = GS.best_estimator_

    cv_scores = pd.DataFrame(GS.cv_results_)

    cv_scores.to_csv(clf_name + '_cv_scores.csv')

    with open(clf_name + '_best_params.txt', 'w+') as f:

        wr = csv.DictWriter(f, fieldnames=list(GS.best_params_.keys()))
        wr.writeheader()
        wr.writerow(GS.best_params_)

    y_pred = clf.predict(X_test)
        
    with open(clf_name + '_report.txt', 'w+') as wr:

        wr.write(classification_report(y_test, y_pred))
        wr.write('\nAccuracy: ' + str(accuracy_score(y_test, y_pred)))
    
    pickle.dump(clf, open(clf_name + '_model.pkl', 'wb+'))
    
    print(clf_name + " Model Saved")

if __name__ == '__main__':

    # Mapping
    d = {'happy':0, 'sad':1, 'angry':2, 'others':3}

    # Loading Vectors
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')  

    # Normalize the data
    X = np.vstack((X_train, X_test))   
    X_train = (X_train - np.mean(X)) / np.std(X)
    X_test = (X_test - np.mean(X)) / np.std(X)

    # Learning SVM
    ########################################################################
    print("\nSVM Model")


    clf = OneVsRestClassifier(LinearSVC(verbose=1, max_iter=30000), n_jobs=-1)

    clf.fit(X_train, y_train)

    Y_pred = clf.predict(X_train)

    train_accuracy = accuracy_score(Y_pred, y_train)

    print('train_accuracy =', train_accuracy)

    Y_pred = clf.predict(X_test)

    test_accuracy = accuracy_score(Y_pred, y_test)

    print('test_accuracy =', test_accuracy)

    pickle.dump(clf, open('LinearSVC_model.pkl', 'wb+'))

    n_estimators = 10

    clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf'), max_samples=1.0 / n_estimators, n_estimators=n_estimators), n_jobs=-1)
    
    param_grid = {'estimator__base_estimator__C':[2**(-5), 2**(0), 2**(5), 2**(10), 2**(15)], 'estimator__base_estimator__gamma':[2**(-15), 2**(-10), 2**(-6), 2**(-2),2**(3)]}
    
    fit_report_save(X_train, X_test, y_train, y_test, clf, 'SVC', param_grid)
    
    # Learning LR
    ##########################################################################
    print("\nLogistic Regression Model")

    clf = LogisticRegression(n_jobs=2)

    param_grid = {'penalty':['l1', 'l2']}

    fit_report_save(X_train, X_test, y_train, y_test, clf, 'LogisticRegression', param_grid)
    
    # Learning NN
    #########################################################################################
    print("\nNeural Net")
    
    clf = MLPClassifier(early_stopping=True)
    
    param_grid = {'hidden_layer_sizes':[(100, 25), (100, 25, 6)], 'solver':['lbfgs', 'adam'], 'alpha':[0.1, 1]}

    fit_report_save(X_train, X_test, y_train, y_test, clf, 'MLPClassifier', param_grid)
    
    # Learning Naive Bayes
    #########################################################################################
    print("\nNaive Bayes")
    
    clf = GaussianNB()
    
    param_grid = {'var_smoothing':[1e-9]}

    fit_report_save(X_train, X_test, y_train, y_test, clf, 'GaussianNB', param_grid)






# def svm(X_train, X_test, y_train, y_test):
#     clf = SVC(gamma='auto')
#     clf.fit(X_train, y_train)

#     print("SVM Trained")

#     pred = clf.predict(X_test)
#     print(classification_report(y_test, pred))
#     pickle.dump(clf, open('svm.sav', 'wb+'))
#     print("Model Saved")

# def knn(X_train, X_test, y_train, y_test):
#     clf = KNeighborsClassifier(n_neighbors=10)
#     clf.fit(X_train, y_train)

#     print("KNN Model Trained")

#     pred = clf.predict(X_test)
#     print(classification_report(y_test, pred))
#     pickle.dump(clf, open('knn.sav', 'wb+'))
#     print("Model Saved")

# def lr(X_train, X_test, y_train, y_test):
#     clf = LogisticRegression()
#     clf.fit(X_train, y_train)

#     print("Logistic Regression Model Trained")

#     pred = clf.predict(X_test)
#     print(classification_report(y_test, pred))
#     pickle.dump(clf, open('lr.sav', 'wb+'))
#     print("Model Saved")

# def nn(X_train, X_test, y_train, y_test):
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, ))
#     clf.fit(X_train, y_train)

#     print("Neural Net Trained")

#     pred = clf.predict(X_test)
#     print(classification_report(y_test, pred))
#     pickle.dump(clf, open('nn.sav', 'wb+'))
#     print("Model Saved")