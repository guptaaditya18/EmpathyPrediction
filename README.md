## Pre-processing of data

1. coverting .csv data to a dataframe
2. Target feature is 'Empathy'. Converting empathy values 1,2,3 to 0 (not very empathetic) and 4,5 to 1 (very empathetic).
3. Removing rows in which value of Empathy is NaN
4. converting categorical variables to one hot encoding
5. converting dataset into dfX and dfY<br>
dfY - contains the target variable-Empathy<br>
dfX - contains all the variables
6. Splitting X and Y into training and testing set
```
import Train, Test, PreProc, importlib
importlib.reload(PreProc)
PreProc.LoadCsv('youngpeoplesurvey/responses.csv')
```
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


```
importlib.reload(PreProc)
PreProc.compSet()
```
```python
def corrSet():

	dfX = pd.read_pickle("./Data/dfX.pkl")
	dfY = pd.read_pickle("./Data/dfY.pkl")

	#finding correlation of 'Empathy' variable with other variables
	correlation = dfX.corrwith(dfY, axis=0, drop=False)

	#finding absolute values of correlation (high negative correlation is also high correlation)
	Corr_abs = correlation.abs()

	# dfX1 = dfX.loc[:,  dfX.corrwith(dfY, axis=0, drop=False) >= 0.1]
	dfX2 = dfX.loc[:,  Corr_abs >= 0.1]

	dfX2.to_pickle("./Data/dfX2.pkl")

	XYSplit("corr")
```
Feature elimination with Correlation
Correlated features: This set uses 36 features which were highly correlated to the target feature ‘Empathy’.

```
importlib.reload(PreProc)
PreProc.corrSet()
```

# Baseline Accuracy
## Random Classification
```
Test.rc()
Test.test("Random Classifier")
accuracy with Random Classifier: 50.495049504950494 %
```
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
​
[[100.          62.22222222]<br>
 [ 82.43243243  56.66666667]<br>
 [ 81.44963145  70.        ]<br>
 [ 77.02702703  63.33333333]<br>
 [ 76.65847666  75.55555556]<br>
 [ 73.46437346  72.22222222]<br>
 [ 75.55282555  71.11111111]<br>
 [ 73.83292383  71.11111111]<br>
 [ 73.83292383  70.        ]<br>
 [ 72.85012285  66.66666667]]<br>
5
![knn](https://user-images.githubusercontent.com/13432475/49681479-f98ad180-fa67-11e8-9fae-d81c5390de6b.png)

KNN model trained


accuracy with K Nearest Neighbors: 69.3069306930693 %

```
Test.test("K Nearest Neighbors")
accuracy with K Nearest Neighbors: 69.3069306930693 %
```

# Random Forest
```
Train.rf(15)
```
​
best max depth : 10
Random Forest model trained

```
Test.test("Random Forest")
```
accuracy with Random Forest: 69.3069306930693 %

# Logistic Regression

Train.lr(10)

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
Best C : 5

SVM model trained

```
Test.test("Support Vector Machine")
```
accuracy with Support Vector Machine: 71.28712871287128 %
