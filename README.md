Pre-processing of data

1. coverting .csv data to a dataframe
​
2. Target feature is 'Empathy'. Converting empathy values 1,2,3 to 0 (not very empathetic) and 4,5 to 1 (very empathetic).
​
3. Removing rows in which value of Empathy is NaN
​
4. converting categorical variables to one hot encoding
​
5. converting dataset into dfX and dfY<br>
dfY - contains the target variable-Empathy<br>
dfX - contains all the variables
​
6. Splitting X and Y into training and testing set

import Train, Test, PreProc, importlib
importlib.reload(PreProc)
PreProc.LoadCsv('youngpeoplesurvey/responses.csv')
C:\Users\gupta\Anaconda3\lib\site-packages\pandas\core\generic.py:5430: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame

See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
  self._update_inplace(new_data)
(1005, 172)
(1005,)
(814, 172)
(814,)
(90, 172)
(90,)
(101, 172)
(101,)

importlib.reload(PreProc)
PreProc.compSet()
(1005, 172)
(1005,)
(814, 172)
(814,)
(90, 172)
(90,)
(101, 172)
(101,)
Feature elimination with Correlation
Correlated features: This set uses 36 features which were highly correlated to the target feature ‘Empathy’.


importlib.reload(PreProc)
PreProc.corrSet()
(1005, 36)
(1005,)
(814, 36)
(814,)
(90, 36)
(90,)
(101, 36)
(101,)
Baseline Accuracy
Random Classification

Test.rc()
Test.test("Random Classifier")
accuracy with Random Classifier: 50.495049504950494 %
Classify to most frequent

Train.mf()
Test.test("Most Frequent")
accuracy with Most Frequent: 67.32673267326733 %
K Nearest Neighbors

importlib.reload(Test)
Train.knn(10)
Test.test("K Nearest Neighbors")
​
[[100.          62.22222222]
 [ 82.43243243  56.66666667]
 [ 81.44963145  70.        ]
 [ 77.02702703  63.33333333]
 [ 76.65847666  75.55555556]
 [ 73.46437346  72.22222222]
 [ 75.55282555  71.11111111]
 [ 73.83292383  71.11111111]
 [ 73.83292383  70.        ]
 [ 72.85012285  66.66666667]]
5
KNN model trained
accuracy with K Nearest Neighbors: 69.3069306930693 %


Test.test("K Nearest Neighbors")
accuracy with K Nearest Neighbors: 69.3069306930693 %
Random Forest

Train.rf(15)
​
[[0.65970516 0.66666667]
 [0.68181818 0.7       ]
 [0.73095823 0.74444444]
 [0.77518428 0.75555556]
 [0.81081081 0.74444444]
 [0.85503686 0.76666667]
 [0.90540541 0.76666667]
 [0.94594595 0.78888889]
 [0.96805897 0.76666667]
 [0.98280098 0.8       ]
 [0.9963145  0.76666667]
 [1.         0.77777778]
 [1.         0.73333333]
 [1.         0.74444444]
 [1.         0.77777778]]
10
Random Forest model trained


Test.test("Random Forest")
accuracy with Random Forest: 69.3069306930693 %
Logistic Regression

Train.lr(10)
​
[[0.76044226 0.72222222]
 [0.75921376 0.72222222]
 [0.76167076 0.71111111]
 [0.76167076 0.71111111]
 [0.75798526 0.71111111]
 [0.75552826 0.71111111]
 [0.75429975 0.71111111]
 [0.75552826 0.71111111]
 [0.75675676 0.71111111]
 [0.75675676 0.71111111]]
1
Logistic Regression model trained


Test.test("Logistic Regression")
accuracy with Logistic Regression: 64.35643564356435 %
Perceptron

Train.perc(100)
​
[[0.72727273 0.72222222]
 [0.67199017 0.66666667]
 [0.35380835 0.34444444]
 [0.71498771 0.73333333]
 [0.66093366 0.66666667]
 [0.71867322 0.67777778]
 [0.67321867 0.67777778]
 [0.73955774 0.72222222]
 [0.7027027  0.65555556]
 [0.66093366 0.66666667]
 [0.72972973 0.71111111]
 [0.72972973 0.71111111]
 [0.67321867 0.66666667]
 [0.73095823 0.74444444]
 [0.68673219 0.7       ]
 [0.72113022 0.72222222]
 [0.44717445 0.35555556]
 [0.71375921 0.7       ]
 [0.73710074 0.75555556]
 [0.66093366 0.66666667]]
95
Perceptron model trained


Test.test("Perceptron")
accuracy with Perceptron: 71.28712871287128 %
Support Vector Machine

Train.sv(10)
[[0.71130221 0.72222222]
 [0.73710074 0.74444444]
 [0.73832924 0.74444444]
 [0.74938575 0.74444444]
 [0.76044226 0.76666667]
 [0.76167076 0.76666667]
 [0.76412776 0.76666667]
 [0.76535627 0.74444444]
 [0.76658477 0.75555556]
 [0.76781327 0.74444444]]
5
SVM model trained


Test.test("Support Vector Machine")
accuracy with Support Vector Machine: 71.28712871287128 %
