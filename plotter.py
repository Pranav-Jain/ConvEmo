import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle

def data_load():

	X_train = np.load('X_train.npy')
	Y_train = np.load('y_train.npy')
	X_test =  np.load('X_test.npy')
	Y_test = np.load('y_test.npy')

    # Normalize the data
	X = np.vstack((X_train, X_test))
	X_train = (X_train - np.mean(X)) / np.std(X)
	X_test = (X_test - np.mean(X)) / np.std(X)

	return X_train, Y_train, X_test, Y_test

def plot_roc(y_test, y_score, clf_name):

	class_map = ['happy', 'sad', 'angry', 'others']

	y_test[np.where(y_test == 3)] = 2
	y_test[np.where(y_test == 4)] = 3

	n_classes = 4
	y = label_binarize(y_test, classes=list(range(n_classes)))

	roc_auc = [0]*n_classes
	fpr = [0]*n_classes
	tpr = [0]*n_classes
	
	for i in range(n_classes):
	    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	plt.figure()
	for i in range(n_classes):
		plt.plot(fpr[i], tpr[i], label=class_map[i] + ' (auc = ' + str(round(roc_auc[i], 2)) + ')')

	plt.plot([0, 1], [0, 1], 'k--', label='reference')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC ' + clf_name)
	plt.legend(loc="lower right")
	plt.show()


def analysis_plots(clf_list):

	X_train, Y_train, X_test, Y_test = data_load()

	for clf_name in clf_list:

		print(clf_name)

		clf = joblib.load(clf_name + '_model.pkl')

		Y_pred = clf.predict(X_train)

		train_accuracy = accuracy_score(Y_pred, Y_train)

		print('train_accuracy =', train_accuracy)

		Y_pred = clf.predict(X_test)

		test_accuracy = accuracy_score(Y_pred, Y_test)

		print('test_accuracy =', test_accuracy)

		with open(clf_name + '_report.txt', 'a+') as wr:
			wr.write('\n\nTrain Accuracy = ' + str(train_accuracy))
			wr.write('\nTest Accuracy = ' +  str(test_accuracy))


		folds = KFold(5)

		train_size, train_score, val_score = learning_curve(clf, X_train, Y_train, cv=folds, n_jobs=-1, verbose=2)

		train_score = np.mean(train_score, axis=1)
		val_score = np.mean(val_score, axis=1)

		plt.figure()
		plt.plot(train_size, train_score, label='Train Accuracy')
		plt.plot(train_size, val_score, label='Validation Accuracy')
		plt.xlabel('Train Size')
		plt.ylabel('Accuracy')
		plt.title(clf_name)
		plt.legend()
		plt.show()

		cm = confusion_matrix(Y_test, Y_pred) 

		cm = (cm.astype('float').T / cm.sum(axis=1)).T
		
		plt.matshow(cm)

		for i in range(cm.shape[0]): 
			for j in range(cm.shape[1]):
				plt.text(j, i, str(round(cm[i, j], 2)), horizontalalignment="center")

		plt.title('Confusion Matrix')
		plt.colorbar()
		plt.ylabel('True class')
		plt.xlabel('Predicted class')
		plt.show()

		y_score = []

		if clf_name == 'MLPClassifier' or clf_name == 'GaussianNB':

			y_score = clf.predict_proba(X_test)

		else :

			y_score = clf.decision_function(X_test)

		plot_roc(Y_test, y_score, clf_name)

if __name__ == '__main__':

	analysis_plots(['GaussianNB', 'MLPClassifier', 'SVC', 'LogisticRegression'])