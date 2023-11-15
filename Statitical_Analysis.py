import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import warnings

data = pd.read_csv("crime.csv")

print(data.head())
print(data.tail())
print(data.info())

print(data.isna().sum())
print(data.columns)

data['THEFT'].plot(kind = 'density')

sns.distplot(data['YEAR'],hist=True,kde=True)