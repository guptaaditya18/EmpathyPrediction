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

Train.knn() :

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

Train.rf():
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
