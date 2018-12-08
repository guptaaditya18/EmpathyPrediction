import pandas as pd
import numpy as np
import youngpeoplesurvey
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler	


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


def compSet():
	XYSplit("raw")


def XYSplit(data):

	if data == "raw":
		dfX = pd.read_pickle("./Data/dfX.pkl")
	elif data == "corr":
		dfX = pd.read_pickle("./Data/dfX2.pkl")

	dfY = pd.read_pickle("./Data/dfY.pkl")

	dfXtrall = dfX.sample(frac = 0.9 , random_state=0)
	dfYtrall = dfY.sample(frac = 0.9, random_state=0)
	dfXte = dfX.drop(dfXtrall.index)
	dfYte = dfY.drop(dfYtrall.index)
	dfXtr = dfXtrall.sample(frac = 0.9, random_state = 0)
	dfYtr = dfYtrall.sample(frac = 0.9, random_state = 0)
	dfXva = dfXtrall.drop(dfXtr.index)
	dfYva = dfYtrall.drop(dfYtr.index)

	dfXtr.to_pickle("./Data/dfXtr.pkl")
	dfYtr.to_pickle("./Data/dfYtr.pkl")
	dfXva.to_pickle("./Data/dfXva.pkl")
	dfYva.to_pickle("./Data/dfYva.pkl")
	dfXte.to_pickle("./Data/dfXte.pkl")
	dfYte.to_pickle("./Data/dfYte.pkl")

	print(dfX.shape)
	print(dfY.shape)
	print(dfXtr.shape)
	print(dfYtr.shape)
	print(dfXva.shape)
	print(dfYva.shape)
	print(dfXte.shape)
	print(dfYte.shape)


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
