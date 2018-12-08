from sklearn.externals import joblib
from Train import *
import numpy as np
import random

def test(model):

	X = pd.read_pickle("./Data/dfXte.pkl")
	Y = pd.read_pickle("./Data/dfYte.pkl")


	if(model == "Random Classifier"):
		randList = []
		for x in range(Y.size):
			randList.append(random.randint(0,1))
		print("accuracy with {}: {} %".format(model, 100 * (np.mean((Y == randList)))))
		return None

	elif(model == "Most Frequent"):
		mostFrequentClass = mf()
		print("accuracy with {}: {} %".format(model, 100 * (np.mean((Y > 0) == mostFrequentClass))))
		return None

	elif(model == "Random Forest"):
		filename = "rf.model"
	elif(model == "Logistic Regression"):
		filename = "lr.model"
	elif(model == "Perceptron"):
		filename = "perc.model"
	elif(model == "Support Vector Machine"):
		filename = "svm.model"
	elif(model == "K Nearest Neighbors"):
		filename = "knn.model"
	else:
		print("Invalid argument, try again")
		return None

	file = "./Models/" + filename
	LoadedModel = joblib.load(file)



	score = LoadedModel.score(X, Y)  
	print("accuracy with {}: {} %".format(model, 100 * score)) 