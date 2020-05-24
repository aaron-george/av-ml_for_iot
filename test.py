import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

df_train=pd.read_csv("train_aWnotuB.csv")
df_test= pd.read_csv("test_BdBKkAj_L87Nc3S.csv")
df_submission=pd.read_csv("sample_submission_KVKNmI7.csv")

df1 = df_train[(df_train['Junction']==1)]
plt.plot(df_train['DateTime'],df_train['Vehicles'])
plt.show()