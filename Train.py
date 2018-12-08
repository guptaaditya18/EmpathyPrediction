from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random



def rc():
	return None

def mf():
	dfXtr = pd.read_pickle("./Data/dfXtr.pkl")
	dfYtr = pd.read_pickle("./Data/dfYtr.pkl")
	mostFrequentClass = None
	ones = 0
	zeros = 0
	for i in range(dfYtr.size):
		if dfYtr.iloc[i] == 0:
			zeros = zeros + 1
		if dfYtr.iloc[i] == 1:
			ones = ones + 1

	if ones >= zeros:
		mostFrequentClass = 1
	else:
		mostFrequentClass = 0

	return mostFrequentClass

def knn(k):
	dfXtr = pd.read_pickle("./Data/dfXtr.pkl")
	dfYtr = pd.read_pickle("./Data/dfYtr.pkl")
	dfXva = pd.read_pickle("./Data/dfXva.pkl")
	dfYva = pd.read_pickle("./Data/dfYva.pkl")
	kiter = k
	x   = np.zeros((kiter,2), dtype=float)

	for i in range(1, kiter+1):
    		knnModel = KNeighborsClassifier(n_neighbors = i)
    		knnModel.fit(dfXtr, dfYtr)
    		x[i-1][0] = knnModel.score(dfXtr, dfYtr)*100
    		x[i-1][1] = knnModel.score(dfXva, dfYva)*100

	y = [i for i in range(1, kiter+1)]
	print(x)

	plt.plot(y, x[:,0], label = 'Test Accuracy', linewidth=2.0)
	plt.plot(y, x[:,1], label = 'Validation Accuracy', linewidth=2.0)
	plt.xlabel('K')
	plt.ylabel('Accuracy (%)')
	plt.title('K Nearest Neighbors')
	plt.legend()
	plt.grid()
	plt.show

	bestk = np.argmax(x[:,1])+1
	print(bestk)
	FinalknnModel = KNeighborsClassifier(n_neighbors = bestk)  
	FinalknnModel.fit(dfXtr, dfYtr)
	joblib.dump(FinalknnModel, "./Models/knn.model")
	print("KNN model trained")

def rf(iter):
	dfXtr = pd.read_pickle("./Data/dfXtr.pkl")
	dfYtr = pd.read_pickle("./Data/dfYtr.pkl")
	dfXva = pd.read_pickle("./Data/dfXva.pkl")
	dfYva = pd.read_pickle("./Data/dfYva.pkl")
	depthiter = iter
	x   = np.zeros((depthiter,2), dtype=float)

	for i in range(1, depthiter+1):
    		RFModel = RandomForestClassifier(max_depth = i, random_state = random.seed(1234), n_estimators = 100)
    		RFModel.fit(dfXtr, dfYtr)
    		x[i-1][0] = RFModel.score(dfXtr, dfYtr)
    		x[i-1][1] = RFModel.score(dfXva, dfYva)

	y = [i for i in range(1, depthiter +1)]
	print(x)

	plt.plot(y, x[:,0], label = 'Test Accuracy', linewidth=2.0)
	plt.plot(y, x[:,1], label = 'Validation Accuracy', linewidth=2.0)
	plt.xlabel('Max depth')
	plt.ylabel('Accuracy (%)')
	plt.title('Random Forest')
	plt.legend()
	plt.grid()
	plt.show

	maxdepth = np.argmax(x[:,1]) + 1
	print(maxdepth)
	FinalRFModel = RandomForestClassifier(max_depth = maxdepth, random_state = random.seed(1234), n_estimators = 100)  
	FinalRFModel.fit(dfXtr, dfYtr)
	joblib.dump(FinalRFModel, "./Models/rf.model")
	print("Random Forest model trained")

def lr(iter):
	dfXtr = pd.read_pickle("./Data/dfXtr.pkl")
	dfYtr = pd.read_pickle("./Data/dfYtr.pkl")
	dfXva = pd.read_pickle("./Data/dfXva.pkl")
	dfYva = pd.read_pickle("./Data/dfYva.pkl")

	cIter = iter
	x   = np.zeros((cIter,2), dtype=float)


	for i in range(1, cIter+1):
    		logReg = LogisticRegression(C = i, random_state = random.seed(1234))
    		logReg.fit(dfXtr, dfYtr)
    		x[i-1][0] = logReg.score(dfXtr, dfYtr)
    		x[i-1][1] = logReg.score(dfXva, dfYva)

	y = [i for i in range(1, cIter+1)]
	print(x)

	plt.plot(y, x[:,0], label = 'Test Accuracy', linewidth=2.0)
	plt.plot(y, x[:,1], label = 'Validation Accuracy', linewidth=2.0)
	plt.xlabel('C')
	plt.ylabel('Accuracy (%)')
	plt.title('Logistic Regression')
	plt.legend()
	plt.grid()
	plt.show

	maxC = np.argmax(x[:,1]) + 1
	print(maxC)
	FinallogReg = LogisticRegression(C = maxC, random_state = random.seed(1234))  
	FinallogReg.fit(dfXtr, dfYtr)
	joblib.dump(FinallogReg, "./Models/lr.model")
	print("Logistic Regression model trained")

def perc(iter):
	dfXtr = pd.read_pickle("./Data/dfXtr.pkl")
	dfYtr = pd.read_pickle("./Data/dfYtr.pkl")
	dfXva = pd.read_pickle("./Data/dfXva.pkl")
	dfYva = pd.read_pickle("./Data/dfYva.pkl")

	depthiter = iter
	size = depthiter//5
	
	x   = np.zeros((size, 2), dtype=float)

	for i in range(5, depthiter+1, 5):
    		PercModel = Perceptron(max_iter=100, random_state = random.seed(1234))
    		PercModel.fit(dfXtr, dfYtr)
    		x[(i//5) - 1][0] = PercModel.score(dfXtr, dfYtr)
    		x[(i//5) - 1][1] = PercModel.score(dfXva, dfYva)

	y = [i for i in range(5, depthiter+1, 5)]
	print(x)

	plt.plot(y, x[:,0], label = 'Test Accuracy', linewidth=2.0)
	plt.plot(y, x[:,1], label = 'Validation Accuracy', linewidth=2.0)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy (%)')
	plt.title('Perceptron')
	plt.legend()
	plt.grid()
	plt.show

	maxiter = (np.argmax(x[:,1]) + 1)*5
	print(maxiter)
	FinalPercModel = Perceptron(max_iter=maxiter, random_state = random.seed(1234))
	FinalPercModel.fit(dfXtr, dfYtr)
	joblib.dump(FinalPercModel, "./Models/perc.model")
	print("Perceptron model trained")

def sv(c):
	dfXtr = pd.read_pickle("./Data/dfXtr.pkl")
	dfYtr = pd.read_pickle("./Data/dfYtr.pkl")
	dfXva = pd.read_pickle("./Data/dfXva.pkl")
	dfYva = pd.read_pickle("./Data/dfYva.pkl")



	citer = c
	x   = np.zeros((citer,2), dtype=float)

	for i in range(1, citer+1):
    		svmModel = svm.SVC(gamma = 0.001, C = i, kernel = 'rbf', random_state = random.seed(1234))
    		svmModel.fit(dfXtr, dfYtr)
    		x[i-1][0] = svmModel.score(dfXtr, dfYtr)
    		x[i-1][1] = svmModel.score(dfXva, dfYva)

	y = [i for i in range(1, citer+1)]
	print(x)

	plt.plot(y, x[:,0], label = 'Test Accuracy', linewidth=2.0)
	plt.plot(y, x[:,1], label = 'Validation Accuracy', linewidth=2.0)
	plt.xlabel('C')
	plt.ylabel('Accuracy (%)')
	plt.title('Support Vector Machine')
	plt.legend()
	plt.grid()
	plt.show

	bestC = np.argmax(x[:,1]) + 1
	print(bestC)
	FinalsvmModel = svm.SVC(gamma = 0.001, C = bestC, kernel = 'rbf', random_state = random.seed(1234))  
	FinalsvmModel.fit(dfXtr, dfYtr)
	joblib.dump(FinalsvmModel, "./Models/svm.model")
	print("SVM model trained")