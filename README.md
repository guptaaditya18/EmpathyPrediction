## Pre-processing of data
Pre-proceesing involved the following steps:
1. coverting .csv data to a dataframe
2. Target feature is 'Empathy'. Converting empathy values 1,2,3 to 0 (not very empathetic) and 4,5 to 1 (very empathetic).
3. Removing rows in which value of Empathy is NaN
4. converting categorical variables to one hot encoding
5. converting dataset into dfX and dfY<br>
dfY - contains the target variable-Empathy<br>
dfX - contains all the variables
6. Splitting X and Y into training and testing set
```python
import Train, Test, PreProc
PreProc.LoadCsv('youngpeoplesurvey/responses.csv')
```
LoadCSV function -
```python
def LoadCsv(filename):


	#coverting .csv data to a dataframe
	dfRaw = pd.read_csv(filename)

	#converting empathy 1,2,3 to 0 (not very empathetic) and 4,5 to 1 (very empathetic)
	dfRaw.loc[(dfRaw['Empathy'] == 1.0) | (dfRaw['Empathy'] == 2.0) | (dfRaw['Empathy'] == 3.0),'Empathy'] = 0
	dfRaw.loc[(dfRaw['Empathy'] == 4.0) | (dfRaw['Empathy'] == 5.0),'Empathy'] = 1

	#Removing rows in which value of Empathy is NaN
	df = dfRaw[pd.isnull(dfRaw.Empathy) == False]

	# Handling NaN values
	df_Mode = df

	for col in df_Mode.columns:
    		df_Mode[col].fillna(df_Mode[col].mode()[0], inplace=True)

    #converting categorical variables to one hot encoding
	dfOneHotMode = pd.get_dummies(df_Mode, dummy_na=False)


	#converting dataset into dfX and dfY

	#dfY - contains the target variable-Empathy
	dfY = dfOneHotMode['Empathy']
	#dfX - contains all the variables
	dfX = dfOneHotMode.drop('Empathy', 1)

	dfX.to_pickle("./Data/dfX.pkl")
	dfY.to_pickle("./Data/dfY.pkl")
	XYSplit("raw")

```

Feature elimination with Correlation
Correlated features: This set uses 36 features which were highly correlated to the target feature ‘Empathy’.

```python
importlib.reload(PreProc)
PreProc.corrSet()
```
```python
def corrSet():

	#finding correlation of 'Empathy' variable with other variables
	correlation = dfX.corrwith(dfY, axis=0, drop=False)

	#finding absolute values of correlation (high negative correlation is also high correlation)
	Corr_abs = correlation.abs()

	# dfX1 = dfX.loc[:,  dfX.corrwith(dfY, axis=0, drop=False) >= 0.1]
	dfX2 = dfX.loc[:,  Corr_abs >= 0.1]

	dfX2.to_pickle("./Data/dfX2.pkl")

	XYSplit("corr")
```

Using compSet function, user can switch back to complete set of features. To switch back to reduced features, user can use corrSet() function again:
```python
importlib.reload(PreProc)
PreProc.compSet()
```

# Baseline Accuracy
## Random Classification
```
Train.rc()
Test.test("Random Classifier")
```

accuracy with Random Classifier: 50.495049504950494 %


## Classify to most frequent
```
Train.mf()
Test.test("Most Frequent")
```
accuracy with Most Frequent: 67.32673267326733 %

# K Nearest Neighbors
```
Train.knn(10)
Test.test("K Nearest Neighbors")
```

![knn](https://user-images.githubusercontent.com/13432475/49681479-f98ad180-fa67-11e8-9fae-d81c5390de6b.png)
Best K : 5
KNN model trained


```
Test.test("K Nearest Neighbors")
```
accuracy with K Nearest Neighbors: 69.3069306930693 %

Train.knn(n) :

parameter - itertare K from 1 to n to find best K based on validation accuracy

function:

```python
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
```


# Random Forest
```
Train.rf(15)
```
![rf](https://user-images.githubusercontent.com/13432475/49681740-61dbb200-fa6c-11e8-919a-3877b8c53358.png)
best max depth : 10
Random Forest model trained

```
Test.test("Random Forest")
```
accuracy with Random Forest: 69.3069306930693 %

Train.rf(n):

parameter - itertare max_depth from 1 to n to find best max_depth based on validation accuracy

function:
```python
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
```

# Logistic Regression

Train.lr(10)
![lr](https://user-images.githubusercontent.com/13432475/49681713-08738300-fa6c-11e8-8d17-d2cad61de45f.png)
Best C : 1
Logistic Regression model trained

```
Test.test("Logistic Regression")
```
accuracy with Logistic Regression: 64.35643564356435 %

Train.lr(n) - 
parameter - itertare C from 1 to n to find best C based on validation accuracy

function:
```python
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
```



#Perceptron
```
Train.perc(100)
```
![perc](https://user-images.githubusercontent.com/13432475/49681733-4670a700-fa6c-11e8-9f0c-6626c567f6b9.png)
Best number of epochs: 95
Perceptron model trained

```python
Test.test("Perceptron")
accuracy with Perceptron: 71.28712871287128 %
```
accuracy with Perceptron: 71.28712871287128 %
Train.perc(n) - 
parameter - itertare epochs from 1 to n to find best number of epochs based on validation accuracy

function:

```python
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
```

# Support Vector Machine
```
Train.sv(10)
```
![rf](https://user-images.githubusercontent.com/13432475/49681740-61dbb200-fa6c-11e8-919a-3877b8c53358.png)
Best C : 5

SVM model trained

```
Test.test("Support Vector Machine")
```
accuracy with Support Vector Machine: 71.28712871287128 %

Train.svm(n) - 
parameter - itertare c from 1 to n to find best C based on validation accuracy

function:

```python
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
```


# Testing

Test(modelName)-
parameter - name of model {"Random Classifier","Most Frequent","Random Forest","Logistic Regression","Support Vector Machine","K Nearest Neighbors"}

function:

```python
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
```

